"""Aurora-ICT launcher — 단일 실행 파일 (.exe / .app).

Aurora launcher와 다른 점:
- 본체가 별도 다운로드 X — 처음부터 PyInstaller로 함께 묶음
- 자동 업데이트는 v0.3.x에서 추가 예정
- pywebview로 http://127.0.0.1:8765/ui/ 띄움
- uvicorn은 같은 프로세스 안에서 background thread로
"""

__version__ = "0.2.3"

__all__ = ["__version__"]
