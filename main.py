from nicegui import ui, events

dark = ui.dark_mode(False)


class Data:
    def __init__(self) -> None:
        self.input_filename = ""
        self.output_filename = "result.yaml"


data = Data()


def process_file(e: events.UploadEventArguments):
    ui.notify(f"Uploaded {e.name}")

    data.input_filename = f"./tmp/input/{e.name}"
    with open(data.input_filename, "wb") as f:
        f.write(e.content.read())


with ui.card().classes("no-shadow w-1/3 fixed-center"):
    with ui.card_section().classes("mx-auto"):
        ui.label("Table Extractor").classes("text-h2")

    with ui.card_section():
        ui.label("Использование:").classes("text-overline")
        ui.label(
            "Выберете PDF файлы, данные из таблиц в которых Вы хотите извлечь"
        ).classes("text-body1")

    with ui.card_section().classes("w-full"):
        ui.upload(
            label="Input file",
            on_upload=process_file,
            auto_upload=True,
        ).props(
            "accept=.pdf"
        ).classes("w-full")

    with ui.card_section().classes("w-full"):
        ui.spinner("bars").classes("w-1/6 h-full mx-auto")

    with ui.card_section().classes("w-full"):
        ui.button(on_click=lambda: ui.download(f"./tmp/output/{data.output_filename}")).bind_text_from(
            data, "output_filename", lambda x: f"Скачать {x}"
        )


ui.run()
