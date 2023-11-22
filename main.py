from nicegui import ui, events
import sys
import os

module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'table_extraction'))
sys.path.insert(0, module_dir)

import extractor

dark = ui.dark_mode(True)


class Data:
    def __init__(self) -> None:
        self.input_filename = ""
        self.output_filename = "result.yaml"

data = Data()

def process_file(e: events.UploadEventArguments):
    """
    Process a file that has been uploaded.
    Args:
        e (events.UploadEventArguments): The event arguments containing information about the uploaded file.
    Returns:
        None
    """
    ui.notify(f"Uploaded {e.name}")

    data.input_filename = f"./tmp/input/{e.name}"
    with open(data.input_filename, "wb") as f:
        f.write(e.content.read())

    # Process the file using the extractor
    output_path = f"./tmp/output/{data.output_filename}"
    extractor.extract(data.input_filename, e.name)


with ui.card().classes("no-shadow w-1/3 fixed-center"):
    with ui.card_section().classes("mx-auto"):
        ui.label("Table Extractor").classes("text-h2").style("color: orange;")

    with ui.card_section():
        ui.label("Использование:").classes("text-overline").style("color: orange;")
        ui.label(
            "Выберете PDF файлы, данные из таблиц в которых Вы хотите извлечь"
        ).classes("text-body1").style("color: orange;")

    with ui.card_section().classes("w-full"):
        ui.upload(
            label="Input file",
            on_upload=process_file,
            auto_upload=True,
        ).props(
            "accept=.pdf"
        ).classes("w-full").style("color: red;")

    with ui.card_section().classes("w-full"):
        ui.spinner("bars").classes("w-1/6 h-full mx-auto")

    with ui.card_section().classes("w-full"):
        ui.button(on_click=lambda: ui.download(f"./tmp/output/{data.output_filename}")).bind_text_from(
            data, "output_filename", lambda x: f"Скачать {x}"
        )

ui.run()
