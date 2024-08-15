import sys
from json_repair import repair_json
import os
import json
import requests
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QFileDialog, 
                             QTextEdit, QRadioButton, QButtonGroup, QListWidget,
                             QSlider, QGroupBox)

def clean_string(data):
    if isinstance(data, dict):
        data = json.dumps(data)
    if isinstance(data, str):
        data = re.sub(r"\n", "", data)
        data = re.sub(r'["""]', '"', data)
        data = re.sub(r"\\{2}", "", data)
        last_period = data.rfind('.')
        if last_period != -1:
            data = data[:last_period+1]
    return data

def clean_json(data):
    if data is None:
        return ""
    if isinstance(data, dict):
        data = json.dumps(data)
        try:
            return json.loads(data)
        except:
            pass
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, data, re.DOTALL)

    if match:
        json_str = match.group(1).strip()
        data = json_str
    else:
        json_str = re.search(r"\{.*\}", data, re.DOTALL)
        if json_str:
            data = json_str.group(0)

    data = re.sub(r"\n", " ", data)
    data = re.sub(r'["""]', '"', data)

    try:
        return json.loads(repair_json(data))
    except json.JSONDecodeError:
        print("JSON error")
    return data

class LLMProcessor:
    def __init__(self, api_url, api_password):
        self.api_url = api_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_password}",
        }
        self.templates = {
            "Alpaca": {"user": "\n\n### Instruction:\n\n", "assistant": "\n\n### Response:\n\n", "system": ""},
            "Vicuna": {"user": "### Human: ", "assistant": "\n### Assistant: ", "system": ""},
            "Llama 2": {"user": "[INST] ", "assistant": " [/INST]", "system": ""},
            "Llama 3": {"endTurn": "<|eot_id|>", "system": "", "user": "<|start_header_id|>user<|end_header_id|>\n\n", "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n"},
            "Phi-3": {"user": "<|end|><|user|>\n", "assistant": "<end_of_turn><|end|><|assistant|>\n", "system": ""},
            "Mistral": {"user": "\n[INST] ", "assistant": " [/INST]\n", "system": ""},
            "Yi": {"user": "<|user|>", "assistant": "<|assistant|>", "system": ""},
            "ChatML": {"user": "<|im_start|>user\n", "assistant": "<|im_end|>\n<|im_start|>assistant\n", "system": ""},
            "WizardLM": {"user": "input:\n", "assistant": "output\n", "system": ""}
        }

    def query_llm(self, template_name, system_instruction, instruction, content, attached_files, sampler_settings):
        template = self.templates[template_name]
        system_part = f"{template['system']}{system_instruction}\n" if system_instruction else ""
        
        full_content = content + "\n\nAttached files:\n"
        for file_name, file_content in attached_files.items():
            full_content += f"[FILE:{file_name}]\n{file_content}\n[/FILE]\n\n"
        end_part = template.get('endTurn', "")
        prompt = f"{system_part}{template['user']}{instruction}\n{full_content}{end_part}{template['assistant']}"
        
        payload = {
            "prompt": prompt,
            "max_length": 1000,
            **sampler_settings
        }
        
        response = requests.post(f"{self.api_url}/api/v1/generate", json=payload, headers=self.headers)
        return {payload, response.json()["results"][0].get("text", "")} 
        
        
class BenchmarkThread(QThread):
    output_received = pyqtSignal(str)

    def __init__(self, llm_processor, template_name, system_instruction, instruction, content, attached_files, sampler_settings, output_file):
        super().__init__()
        self.llm_processor = llm_processor
        self.template_name = template_name
        self.system_instruction = system_instruction
        self.instruction = instruction
        self.content = content
        self.attached_files = attached_files
        self.sampler_settings = sampler_settings
        self.output_file = output_file

    def run(self):
        try:
            response = self.llm_processor.query_llm(
                self.template_name, self.system_instruction, self.instruction, 
                self.content, self.attached_files, self.sampler_settings
            )
            
            with open(f"{self.output_file}_structured.json", "w") as f:
                json.dump(response, f, indent=2)
            
            with open(f"{self.output_file}_plain.txt", "w") as f:
                f.write(str(response))
            
            self.output_received.emit(f"Response saved to {self.output_file}_structured.json and {self.output_file}_plain.txt")
        except Exception as e:
            self.output_received.emit(f"Error: {str(e)}")

class SamplerSlider(QWidget):
    def __init__(self, name, min_value, max_value, step):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.name = name
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(min_value)
        self.slider.setMaximum(max_value)
        self.slider.setSingleStep(step)
        self.slider.setValue(min_value)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        
        self.label = QLabel(f"{name}: Off")
        layout.addWidget(self.label)
        layout.addWidget(self.slider)

        self.slider.valueChanged.connect(self.update_label)

    def update_label(self, value):
        if value == self.slider.minimum():
            self.label.setText(f"{self.name}: Off")
        else:
            self.label.setText(f"{self.name}: {value / 100:.2f}")

    def value(self):
        if self.slider.value() == self.slider.minimum():
            return None
        return self.slider.value() / 100

class LLMBenchmarkGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Benchmark GUI")
        self.setGeometry(100, 100, 800, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # API URL and Password
        api_layout = QHBoxLayout()
        self.api_url_input = QLineEdit("http://localhost:5001")
        self.api_password_input = QLineEdit()
        api_layout.addWidget(QLabel("API URL:"))
        api_layout.addWidget(self.api_url_input)
        api_layout.addWidget(QLabel("API Password:"))
        api_layout.addWidget(self.api_password_input)
        layout.addLayout(api_layout)

        # Prompt template selection
        template_layout = QHBoxLayout()
        self.template_group = QButtonGroup(self)
        for template_name in ["Mistral", "Vicuna", "Llama 3", "ChatML", "Phi-3", "Yi", "WizardLM", "Alpaca"]:
            radio = QRadioButton(template_name)
            self.template_group.addButton(radio)
            template_layout.addWidget(radio)
        layout.addLayout(template_layout)

        # System instruction
        self.system_instruction_input = QLineEdit()
        layout.addWidget(QLabel("System Instruction:"))
        layout.addWidget(self.system_instruction_input)

        # Instruction
        self.instruction_input = QTextEdit()
        layout.addWidget(QLabel("Instruction:"))
        layout.addWidget(self.instruction_input)

        # Content
        self.content_input = QTextEdit()
        layout.addWidget(QLabel("Content:"))
        layout.addWidget(self.content_input)

        # File upload
        file_layout = QHBoxLayout()
        self.file_list = QListWidget()
        self.upload_button = QPushButton("Upload Files")
        self.upload_button.clicked.connect(self.upload_files)
        file_layout.addWidget(self.file_list)
        file_layout.addWidget(self.upload_button)
        layout.addLayout(file_layout)

        # Sampler settings
        sampler_group = QGroupBox("Sampler Settings")
        sampler_layout = QVBoxLayout()
        self.top_k_slider = SamplerSlider("Top K", 0, 100, 1)
        self.top_p_slider = SamplerSlider("Top P", 0, 100, 1)
        self.min_p_slider = SamplerSlider("Min P", 0, 100, 1)
        self.temperature_slider = SamplerSlider("Temperature", 0, 200, 1)
        
        sampler_layout.addWidget(self.top_k_slider)
        sampler_layout.addWidget(self.top_p_slider)
        sampler_layout.addWidget(self.min_p_slider)
        sampler_layout.addWidget(self.temperature_slider)
        sampler_group.setLayout(sampler_layout)
        layout.addWidget(sampler_group)

        # Output file
        output_layout = QHBoxLayout()
        self.output_file_input = QLineEdit()
        output_button = QPushButton("Select Output File")
        output_button.clicked.connect(self.select_output_file)
        output_layout.addWidget(QLabel("Output File:"))
        output_layout.addWidget(self.output_file_input)
        output_layout.addWidget(output_button)
        layout.addLayout(output_layout)

        # Run button
        self.run_button = QPushButton("Run Benchmark")
        self.run_button.clicked.connect(self.run_benchmark)
        layout.addWidget(self.run_button)

        # Output area
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        layout.addWidget(QLabel("Output:"))
        layout.addWidget(self.output_area)

        self.attached_files = {}

    def select_output_file(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Select Output File")
        if file_name:
            self.output_file_input.setText(file_name)

    def upload_files(self):
        file_names, _ = QFileDialog.getOpenFileNames(self, "Select Files to Upload")
        for file_name in file_names:
            with open(file_name, 'rb') as file:
                file_content = base64.b64encode(file.read()).decode('utf-8')
                self.attached_files[os.path.basename(file_name)] = file_content
                self.file_list.addItem(os.path.basename(file_name))

    def run_benchmark(self):
        api_url = self.api_url_input.text()
        api_password = self.api_password_input.text()
        template_name = self.template_group.checkedButton().text()
        system_instruction = self.system_instruction_input.text()
        instruction = self.instruction_input.toPlainText()
        content = self.content_input.toPlainText()
        output_file = self.output_file_input.text()

        sampler_settings = {
            "top_k": self.top_k_slider.value(),
            "top_p": self.top_p_slider.value(),
            "min_p": self.min_p_slider.value(),
            "temperature": self.temperature_slider.value(),
        }
        sampler_settings = {k: v for k, v in sampler_settings.items() if v is not None}

        llm_processor = LLMProcessor(api_url, api_password)
        self.benchmark_thread = BenchmarkThread(
            llm_processor, template_name, system_instruction, instruction, 
            content, self.attached_files, sampler_settings, output_file
        )
        self.benchmark_thread.output_received.connect(self.update_output)
        self.benchmark_thread.start()

        self.output_area.clear()
        self.output_area.append("Running benchmark...\n")
        self.run_button.setEnabled(False)

    def update_output(self, text):
        self.output_area.append(text)
        self.output_area.verticalScrollBar().setValue(self.output_area.verticalScrollBar().maximum())
        QApplication.processEvents()
        if "Error" in text or "Response saved" in text:
            self.run_button.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LLMBenchmarkGUI()
    window.show()
    sys.exit(app.exec())
