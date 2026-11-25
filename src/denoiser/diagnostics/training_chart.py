"""Run directly or with --run layers_7_channels_64_v001 --no-browser.

Reads completed epochs from metrics.csv and Gaussian rows from quality.csv.
No training imports, checkpoint writes, or additional Python dependencies.
"""

import argparse
import csv
import io
import json
import math
import runpy
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


# Paths and server.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIRECTORY = PROJECT_ROOT / "data" / "result"
DEFAULT_RUN = ""  # Empty: follow the newest run. Or set a result folder name.
HOST = "127.0.0.1"
PORT = 8765
POLL_INTERVAL_MS = 1500
OPEN_BROWSER = True
ECHARTS_FILE = Path(__file__).with_name("vendor") / "echarts.min.js"
ECHARTS_URL = "https://cdn.jsdelivr.net/npm/echarts@6.0.0/dist/echarts.min.js"

# All chart text, colors, and animation controls.
PAGE_TITLE = "Denoiser · Training Observatory"
AXIS_EPOCH = "Эпоха"
FIRST_EPOCH = 1
AXIS_PSNR = "PSNR · dB"
AXIS_GAIN = "Gain · dB"
AXIS_LOSS = "Loss · L1 / MSE"
PANEL_TITLES = ["01  КАЧЕСТВО ВОССТАНОВЛЕНИЯ", "02  GAUSSIAN · УЛУЧШЕНИЕ", "03  ОШИБКА ОБУЧЕНИЯ"]
COLORS = {
    "background": "#090e1a", "surface": "#101929", "text": "#e5edf8",
    "muted": "#8293ad", "grid": "#203049", "train": "#67e8c5",
    "eval": "#8ea7ff", "train_gain": "#ffd07b", "eval_gain": "#ed9cff",
    "loss": "#ff8d9e",
}
SERIES = [
    {"key": "train_psnr", "name": "Train PSNR", "panel": 0, "color": COLORS["train"]},
    {"key": "eval_psnr", "name": "Eval PSNR", "panel": 0, "color": COLORS["eval"]},
    {"key": "train_gain", "name": "Train Gaussian gain", "panel": 1, "color": COLORS["train_gain"]},
    {"key": "eval_gain", "name": "Eval Gaussian gain", "panel": 1, "color": COLORS["eval_gain"]},
    {"key": "loss", "name": "Train loss", "panel": 2, "color": COLORS["loss"]},
]
DRAW_DURATION_MS = 1500
UPDATE_DURATION_MS = 850
ANIMATION_EASING = "cubicOut"
PULSE_DURATION_MS = 2200
PULSE_EASING = "ease-in-out"
MARKER_MOVE_EASING = "ease-out"
PULSE_MIN_SCALE = 0.8
PULSE_MAX_SCALE = 1.35
PULSE_MIN_OPACITY = 0.7
POINT_SIZE_PX = 9
HISTORY_POINT_SIZE_PX = 5
LINE_WIDTH_PX = 2.4
LINE_GLOW_PX = 12
POINT_GLOW_PX = 18
LINE_SMOOTHING = False  # Preserve measured values without curved overshoot.
CHART_HEIGHT_PX = 850


Messages = runpy.run_path(str(PROJECT_ROOT / "src/denoiser/utils/errors.py"))["DiagnosticsMessages"]


def number(value):
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def read_csv(path):
    if not path.exists():
        return []

    before = path.stat()
    content = path.read_text(encoding="utf-8-sig")
    after = path.stat()

    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OSError(Messages.FILE_CHANGING)

    return [row for row in csv.DictReader(io.StringIO(content))
            if None not in row and all(value is not None for value in row.values())]


class Reports:
    def __init__(self, fixed_run=None):
        self.fixed_run = fixed_run
        self.cache = {}
        self.generations = {}
        self.lock = threading.Lock()

    def runs(self):
        if self.fixed_run is not None:
            return [self.fixed_run]

        if not RESULTS_DIRECTORY.exists():
            return []

        return sorted((path for path in RESULTS_DIRECTORY.iterdir()
                       if path.is_dir() and ((path / "config.yaml").exists() or (path / "metrics.csv").exists())),
                      key=lambda path: (path / "config.yaml").stat().st_mtime_ns
                      if (path / "config.yaml").exists() else path.stat().st_mtime_ns, reverse=True)

    def snapshot(self, selected=""):
        with self.lock:
            runs = self.runs()
            names = [path.name for path in runs]
            directory = next((path for path in runs if path.name == selected), None) if selected else next(iter(runs), None)

            if directory is None:
                return {"run": "", "runs": names, "rows": [], "diagnostic": False}

            try:
                folder_stat = directory.stat()
                generation = f"{folder_stat.st_ino}:{getattr(folder_stat, 'st_birthtime_ns', 0)}"

                if self.generations.get(directory) != generation:
                    self.cache.pop(directory, None)
                    self.generations[directory] = generation

                metrics = read_csv(directory / "metrics.csv")
                quality = read_csv(directory / "quality.csv")
                gaussian = {(row["epoch"], row["stage"]): number(row.get("gain_db"))
                            for row in quality if row.get("noise") == "gaussian"
                            and "epoch" in row and "stage" in row}
                rows = {}

                for row in metrics:
                    epoch = number(row.get("epoch"))

                    if epoch is None or epoch < 1 or epoch != int(epoch) or number(row.get("loss")) is None:
                        continue

                    rows[int(epoch)] = {
                        "epoch": int(epoch), "loss": number(row.get("loss")),
                        "train_psnr": number(row.get("train_psnr_db")),
                        "eval_psnr": number(row.get("eval_psnr_db")),
                        "train_gain": gaussian.get((row["epoch"], "train")),
                        "eval_gain": gaussian.get((row["epoch"], "eval")),
                        "lr": number(row.get("lr")),
                        "eval_image": (directory / f"eval_epoch_{int(epoch):05d}.jpg").is_file(),
                    }

                # The trainer rewrites CSV files; keep the last valid snapshot during a write.
                previous = self.cache.get(directory)

                if previous and len(rows) < len(previous["rows"]):
                    return {**previous, "runs": names}

                report = directory / "metrics.md"
                diagnostic = report.exists() and "DIAGNOSTIC:" in report.read_text(encoding="utf-8")
                result = {"run": directory.name, "run_id": generation, "runs": names,
                          "rows": [rows[key] for key in sorted(rows)], "diagnostic": diagnostic}
                self.cache[directory] = result
                return result
            except (OSError, UnicodeError, csv.Error, KeyError, ValueError):
                return {**self.cache.get(directory, {"run": directory.name, "rows": [], "diagnostic": False}),
                        "runs": names, "reading": True}


def page():
    config = {"title": PAGE_TITLE, "colors": COLORS, "series": SERIES,
              "axis": [AXIS_PSNR, AXIS_GAIN, AXIS_LOSS], "epoch": AXIS_EPOCH, "firstEpoch": FIRST_EPOCH,
              "panels": PANEL_TITLES, "poll": POLL_INTERVAL_MS,
              "draw": DRAW_DURATION_MS, "update": UPDATE_DURATION_MS,
              "easing": ANIMATION_EASING, "smooth": LINE_SMOOTHING,
              "libraryError": Messages.LIBRARY_FAILED, "connectionError": Messages.CONNECTION_FAILED,
              "imageMissing": Messages.EVAL_IMAGE_MISSING, "pointSize": HISTORY_POINT_SIZE_PX}
    replacements = {
        "__CONFIG__": json.dumps(config, ensure_ascii=False).replace("<", "\\u003c"),
        "__HEIGHT__": str(CHART_HEIGHT_PX), "__PULSE__": str(PULSE_DURATION_MS),
        "__MIN_SCALE__": str(PULSE_MIN_SCALE), "__MAX_SCALE__": str(PULSE_MAX_SCALE),
        "__POINT_SIZE__": str(POINT_SIZE_PX), "__POINT_GLOW__": str(POINT_GLOW_PX),
        "__LINE_WIDTH__": str(LINE_WIDTH_PX), "__LINE_GLOW__": str(LINE_GLOW_PX),
        "__PULSE_EASING__": PULSE_EASING, "__MOVE_EASING__": MARKER_MOVE_EASING,
        "__MIN_OPACITY__": str(PULSE_MIN_OPACITY),
    }
    html = HTML

    for key, value in replacements.items():
        html = html.replace(key, value)

    return html.encode("utf-8")


HTML = r'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Training Observatory</title>
<style>
*{box-sizing:border-box}body{margin:0;font:14px/1.5 "Segoe UI",sans-serif;background:var(--background);color:var(--text)}
main{max-width:1440px;margin:auto;padding:36px 32px 24px}header{display:flex;justify-content:space-between;align-items:center;gap:20px;flex-wrap:wrap}
.eyebrow{font-size:11px;letter-spacing:3px;color:var(--train);font-weight:700}h1{font-size:30px;font-weight:600;letter-spacing:-1px;margin:5px 0}
.subtitle,footer{color:var(--muted)}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
select,button{color:var(--text);background:var(--surface);border:1px solid var(--grid);padding:11px 15px;border-radius:12px;font:inherit;max-width:100%}
button{cursor:pointer}button:hover{border-color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin:24px 0}
.card{border:1px solid var(--grid);border-radius:18px;padding:17px 20px;background:linear-gradient(140deg,var(--surface),var(--background))}
.label{font-size:11px;color:var(--muted);letter-spacing:1px}.value{font-size:27px;font-variant-numeric:tabular-nums;margin-top:6px;letter-spacing:-.7px}
.chart-shell{position:relative;border:1px solid var(--grid);border-radius:24px;background:var(--surface);overflow:hidden}
#chart{height:__HEIGHT__px;width:100%}#markers{position:absolute;inset:0;pointer-events:none;overflow:hidden}
.marker{position:absolute;display:none;transition:left __UPDATE_DURATION__ms __MOVE_EASING__,top __UPDATE_DURATION__ms __MOVE_EASING__}
.marker span{position:absolute;left:0;top:0;width:__POINT_SIZE__px;height:__POINT_SIZE__px;border-radius:50%;background:currentColor;box-shadow:0 0 __POINT_GLOW__px currentColor;animation:pulse __PULSE__ms __PULSE_EASING__ infinite}
@keyframes pulse{0%,100%{transform:translate(-50%,-50%) scale(__MIN_SCALE__);opacity:__MIN_OPACITY__}50%{transform:translate(-50%,-50%) scale(__MAX_SCALE__);opacity:1}}
.status{font-size:12px;color:var(--muted)}.notice{display:none;margin:14px 0;color:var(--train_gain)}footer{display:flex;justify-content:space-between;gap:15px;flex-wrap:wrap;font-size:12px;margin-top:16px}
@media(max-width:800px){main{padding:20px 12px}.cards{grid-template-columns:repeat(2,1fr)}.value{font-size:23px}h1{font-size:25px}}
@media(prefers-reduced-motion:reduce){.marker span{animation:none}.marker{transition:none}}
</style></head><body><main>
<header><div><div class="eyebrow">NEURAL NETWORK / LIVE METRICS</div><h1>Training Observatory</h1>
<div class="subtitle">PSNR · Gaussian gain · loss</div></div><div class="controls"><select id="runs" aria-label="Запуск"><option value="">Последний запуск · автоматически</option></select><button id="reset">Сбросить масштаб</button></div></header>
<div class="cards" id="cards"></div><div id="notice" class="notice">Диагностический запуск: train/test могут пересекаться. Eval не измеряет обобщение.</div>
<div class="chart-shell"><div id="chart"></div><div id="markers"></div></div>
<footer><span id="status" class="status">Подключение…</span><span>Точка: JPEG эпохи · Колесо: масштаб · PNG: кнопка в графике</span></footer>
</main><script>const C=__CONFIG__;
for(const [key,value] of Object.entries(C.colors))document.documentElement.style.setProperty('--'+key,value);
document.title=C.title;
</script><script src="/echarts.min.js"></script><script>
const status=document.getElementById('status'), select=document.getElementById('runs');
if(!window.echarts){status.textContent=C.libraryError;throw new Error(C.libraryError)}
const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
const chart=echarts.init(document.getElementById('chart'),null,{renderer:'canvas'});
let rows=[],currentRun='',currentRunId='',signature='',selectedLegend={},generation=0;
const markerNodes=C.series.map(s=>{const node=document.createElement('div');node.className='marker';node.style.color=s.color;node.innerHTML='<span></span>';document.getElementById('markers').append(node);return node});
const cards=document.getElementById('cards');
for(const s of C.series){const card=document.createElement('div');card.className='card';const label=document.createElement('div');label.className='label';label.textContent=s.name;const value=document.createElement('div');value.className='value';value.style.color=s.color;value.textContent='—';card.append(label,value);cards.append(card)}
const grids=[{left:85,right:40,top:100,height:190},{left:85,right:40,top:370,height:160},{left:85,right:40,top:610,bottom:90}];
const axisStyle={axisLine:{lineStyle:{color:C.colors.grid}},axisTick:{show:false},axisLabel:{color:C.colors.muted},nameTextStyle:{color:C.colors.muted},splitLine:{lineStyle:{color:C.colors.grid,type:'dashed'}}};
chart.setOption({backgroundColor:'transparent',textStyle:{fontFamily:'Segoe UI, sans-serif'},animation:!reduced,
animationDuration:C.draw,animationDurationUpdate:C.update,animationEasing:C.easing,animationEasingUpdate:C.easing,
title:C.panels.map((text,i)=>({text,top:[65,335,575][i],left:28,textStyle:{fontSize:11,fontWeight:600,color:C.colors.muted}})),
legend:{top:18,left:24,right:65,textStyle:{color:C.colors.text},icon:'roundRect',itemWidth:15,itemHeight:4,data:C.series.map(s=>s.name)},
toolbox:{right:18,top:12,iconStyle:{borderColor:C.colors.muted},feature:{saveAsImage:{title:'Сохранить PNG',backgroundColor:C.colors.background},dataZoom:{title:{zoom:'Выбрать область',back:'Назад'},yAxisIndex:'none'}}},
tooltip:{trigger:'axis',backgroundColor:C.colors.background,borderColor:C.colors.grid,textStyle:{color:C.colors.text},valueFormatter:value=>value==null?'—':Number(value).toFixed(4)},
axisPointer:{link:[{xAxisIndex:'all'}]},grid:grids,
xAxis:grids.map((_,i)=>({...axisStyle,type:'value',gridIndex:i,min:C.firstEpoch,minInterval:1,name:i===2?C.epoch:'',nameLocation:'middle',nameGap:30,splitLine:{show:false}})),
yAxis:grids.map((_,i)=>({...axisStyle,type:'value',gridIndex:i,name:C.axis[i],nameLocation:'middle',nameRotate:90,nameGap:55,scale:true,min:i===2?0:undefined})),
dataZoom:[{id:'inside',type:'inside',xAxisIndex:[0,1,2],filterMode:'none',zoomOnMouseWheel:true,moveOnMouseMove:true},{id:'slider',type:'slider',xAxisIndex:[0,1,2],bottom:14,height:22,borderColor:C.colors.grid,textStyle:{color:C.colors.muted},fillerColor:C.colors.eval+'18',handleStyle:{color:C.colors.eval},dataBackground:{lineStyle:{color:C.colors.muted},areaStyle:{color:C.colors.grid}}}],
series:C.series.map(s=>({id:s.key,name:s.name,type:'line',xAxisIndex:s.panel,yAxisIndex:s.panel,showSymbol:true,symbol:'circle',symbolSize:C.pointSize,smooth:C.smooth,connectNulls:false,clip:true,
lineStyle:{color:s.color,width:__LINE_WIDTH__,shadowColor:s.color,shadowBlur:__LINE_GLOW__},itemStyle:{color:s.color},emphasis:{focus:'series'},data:[],
markLine:s.key==='train_gain'?{silent:true,symbol:'none',label:{show:false},lineStyle:{color:C.colors.muted,type:'dashed',opacity:.5},data:[{yAxis:0}]}:undefined}))});
function positionMarkers(){C.series.forEach((s,i)=>{const tail=[...rows].reverse().find(r=>r[s.key]!=null),node=markerNodes[i];if(!tail||selectedLegend[s.name]===false){node.style.display='none';return}const pixel=chart.convertToPixel({xAxisIndex:s.panel,yAxisIndex:s.panel},[tail.epoch,tail[s.key]]);if(!pixel||!chart.containPixel({gridIndex:s.panel},pixel)){node.style.display='none';return}node.style.display='block';node.style.left=pixel[0]+'px';node.style.top=pixel[1]+'px'})}
chart.on('rendered',positionMarkers);
chart.on('click','series',event=>{const epoch=event.value?.[0];const row=rows.find(value=>value.epoch===epoch);if(!row?.eval_image){status.textContent=C.imageMissing;return}window.open('/eval-image?run='+encodeURIComponent(currentRun)+'&epoch='+epoch,'_blank','noopener')});
chart.on('legendselectchanged',event=>{selectedLegend=event.selected;positionMarkers()});
new ResizeObserver(()=>{chart.resize();positionMarkers()}).observe(document.getElementById('chart'));
document.getElementById('reset').onclick=()=>chart.dispatchAction({type:'dataZoom',start:0,end:100});
select.onchange=()=>{signature='';generation++;poll()};
async function poll(){const requested=select.value,token=generation;try{const response=await fetch('/api/metrics?run='+encodeURIComponent(requested),{cache:'no-store'});if(!response.ok)throw new Error(C.connectionError);const payload=await response.json();if(token!==generation)return;
const available=JSON.stringify(payload.runs);if(select.dataset.runs!==available){select.replaceChildren(new Option('Последний запуск · автоматически',''),...payload.runs.map(name=>new Option(name,name)));select.value=requested;select.dataset.runs=available}
const nextSignature=JSON.stringify([payload.run,payload.run_id,payload.rows]);
if(nextSignature!==signature){const switched=currentRunId!==payload.run_id;currentRun=payload.run;currentRunId=payload.run_id;rows=payload.rows;signature=nextSignature;
if(switched){chart.dispatchAction({type:'dataZoom',start:0,end:100})}
chart.setOption({series:C.series.map(s=>({id:s.key,data:rows.map(r=>({name:String(r.epoch),value:[r.epoch,r[s.key]]}))}))});
const last=rows.at(-1);document.querySelectorAll('.card .value').forEach((node,i)=>{const value=last?.[C.series[i].key];node.textContent=value==null?'—':value.toFixed(C.series[i].key==='loss'?5:3)})}
document.getElementById('notice').style.display=payload.diagnostic?'block':'none';
status.textContent=(payload.run||'Нет запусков')+' · '+(rows.length?'Эпоха '+rows.at(-1).epoch:'Ожидание завершённой эпохи')+' · '+(payload.reading?'Записывается отчёт…':'Проверено '+new Date().toLocaleTimeString());
}catch(error){status.textContent=C.connectionError+' · '+error.message}}
async function loop(){await poll();setTimeout(loop,C.poll)}loop();
</script></body></html>'''.replace('__UPDATE_DURATION__', str(UPDATE_DURATION_MS))


def handler(reports):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)

            if url.path == "/":
                self.respond(page(), "text/html; charset=utf-8")
            elif url.path == "/api/metrics":
                selected = parse_qs(url.query).get("run", [""])[0]
                payload = reports.snapshot(selected)
                self.respond(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                             "application/json; charset=utf-8")
            elif url.path == "/echarts.min.js":
                if ECHARTS_FILE.exists():
                    self.respond(ECHARTS_FILE.read_bytes(), "text/javascript; charset=utf-8")
                else:
                    self.send_response(302)
                    self.send_header("Location", ECHARTS_URL)
                    self.end_headers()
            elif url.path == "/eval-image":
                query = parse_qs(url.query)
                selected = query.get("run", [""])[0]
                value = query.get("epoch", [""])[0]
                directory = next((path for path in reports.runs() if path.name == selected), None)

                if directory is None or not value.isascii() or not value.isdigit() or len(value) > 9 or int(value) < 1:
                    self.send_error(404, Messages.EVAL_IMAGE_MISSING)
                    return

                path = directory / f"eval_epoch_{int(value):05d}.jpg"

                if not path.is_file() or path.resolve().parent != directory.resolve():
                    self.send_error(404, Messages.EVAL_IMAGE_MISSING)
                    return

                try:
                    content = path.read_bytes()
                except OSError:
                    self.send_error(404, Messages.EVAL_IMAGE_MISSING)
                    return

                self.respond(content, "image/jpeg")
            else:
                self.send_error(404, Messages.NOT_FOUND)

        def respond(self, content, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN, help="Result folder name or absolute path; default: newest run")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    directory = None

    if args.run:
        directory = Path(args.run)
        directory = (RESULTS_DIRECTORY / directory).resolve() if not directory.is_absolute() else directory.resolve()

        if not directory.is_dir():
            parser.error(Messages.RUN_NOT_FOUND.format(path=directory))

    if not 1 <= args.port <= 65535:
        parser.error(Messages.INVALID_PORT)

    reports = Reports(directory)
    server = ThreadingHTTPServer((HOST, args.port), handler(reports))
    address = f"http://{HOST}:{args.port}"
    print(f"{PAGE_TITLE}: {address}\nReports: {directory or RESULTS_DIRECTORY}\nStop: Ctrl+C", flush=True)

    if OPEN_BROWSER and not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(address)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()