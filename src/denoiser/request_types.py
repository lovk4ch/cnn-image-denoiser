from typing import Annotated

from fastapi import Form


Iterations = Annotated[int, Form(ge=1, le=3)]