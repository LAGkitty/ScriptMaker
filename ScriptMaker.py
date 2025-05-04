import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import re
import json
import random
import threading
import queue
import datetime
import os
import sys

# Optional imports - don't fail if not available
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    # Try to import local AI modules
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    HAS_LOCAL_AI = True
except ImportError:
    HAS_LOCAL_AI = False

class ScriptMakerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Script Maker GUI")
        self.root.geometry("800x600")
        self.root.minsize(600, 500)
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Can be: clam, alt, default, classic
        
        # Main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Mode selection
        mode_frame = ttk.Frame(main_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(mode_frame, text="Translation Mode:").pack(side=tk.LEFT, padx=5)
        self.mode_var = tk.StringVar(value="Rule-Based")
        modes = ["Rule-Based", "AI-Powered (Online)", "AI-Powered (Offline)"]
        mode_dropdown = ttk.Combobox(mode_frame, textvariable=self.mode_var, values=modes, width=20)
        mode_dropdown.pack(side=tk.LEFT, padx=5)
        mode_dropdown.bind("<<ComboboxSelected>>", self.on_mode_change)
        
        # AI Settings Frame (initially hidden)
        self.ai_settings_frame = ttk.LabelFrame(main_frame, text="AI Settings")
        
        # API Key Entry (for online AI)
        self.online_ai_frame = ttk.Frame(self.ai_settings_frame)
        self.online_ai_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.online_ai_frame, text="OpenAI API Key:").pack(side=tk.LEFT, padx=5)
        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(self.online_ai_frame, textvariable=self.api_key_var, width=40, show="*")
        self.api_key_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Online AI model selection
        model_frame = ttk.Frame(self.online_ai_frame)
        model_frame.pack(fill=tk.X, pady=5)
        ttk.Label(model_frame, text="AI Model:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="gpt-3.5-turbo")
        models = ["gpt-3.5-turbo", "gpt-4"]
        model_dropdown = ttk.Combobox(model_frame, textvariable=self.model_var, values=models, width=15)
        model_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Test API Connection button
        ttk.Button(self.online_ai_frame, text="Test API Connection", command=self.test_api_connection).pack(anchor=tk.E, padx=5, pady=5)
        
        # Local AI model settings
        self.local_ai_frame = ttk.Frame(self.ai_settings_frame)
        
        # Local model selection
        local_model_frame = ttk.Frame(self.local_ai_frame)
        local_model_frame.pack(fill=tk.X, pady=5)
        ttk.Label(local_model_frame, text="Local Model:").pack(side=tk.LEFT, padx=5)
        self.local_model_var = tk.StringVar(value="TinyLlama")
        local_models = ["TinyLlama", "Phi-2", "GPTQ", "Custom"]
        local_model_dropdown = ttk.Combobox(local_model_frame, textvariable=self.local_model_var, values=local_models, width=15)
        local_model_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Custom model path
        custom_model_frame = ttk.Frame(self.local_ai_frame)
        custom_model_frame.pack(fill=tk.X, pady=5)
        ttk.Label(custom_model_frame, text="Custom Model Path:").pack(side=tk.LEFT, padx=5)
        self.custom_model_var = tk.StringVar()
        custom_model_entry = ttk.Entry(custom_model_frame, textvariable=self.custom_model_var, width=30)
        custom_model_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(custom_model_frame, text="Browse...", 
                   command=lambda: self.custom_model_var.set(filedialog.askdirectory())).pack(side=tk.RIGHT, padx=5)
        
        # Device selection
        device_frame = ttk.Frame(self.local_ai_frame)
        device_frame.pack(fill=tk.X, pady=5)
        ttk.Label(device_frame, text="Device:").pack(side=tk.LEFT, padx=5)
        self.device_var = tk.StringVar(value="CPU")
        devices = ["CPU"]
        if HAS_LOCAL_AI and torch.cuda.is_available():
            devices.append("CUDA (GPU)")
        device_dropdown = ttk.Combobox(device_frame, textvariable=self.device_var, values=devices, width=15)
        device_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Model status and installation button
        model_status_frame = ttk.Frame(self.local_ai_frame)
        model_status_frame.pack(fill=tk.X, pady=5)
        self.model_status_var = tk.StringVar(value="Status: Model not loaded")
        ttk.Label(model_status_frame, textvariable=self.model_status_var).pack(side=tk.LEFT, padx=5)
        ttk.Button(model_status_frame, text="Download/Setup Model", 
                   command=self.setup_local_model).pack(side=tk.RIGHT, padx=5)
        
        # Check if local AI is available
        if not HAS_LOCAL_AI:
            local_ai_warning = ttk.Label(self.local_ai_frame, 
                                       text="Local AI modules not found. Install with:\npip install transformers torch", 
                                       foreground="red")
            local_ai_warning.pack(pady=10)
        
        # Language selection
        lang_frame = ttk.Frame(main_frame)
        lang_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(lang_frame, text="Target Language:").pack(side=tk.LEFT, padx=5)
        self.lang_var = tk.StringVar(value="Lua")
        languages = ["Lua", "Python", "JavaScript", "C#", "Java"]
        lang_dropdown = ttk.Combobox(lang_frame, textvariable=self.lang_var, values=languages, width=15)
        lang_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Input area
        input_frame = ttk.LabelFrame(main_frame, text="Script Commands (one per line)")
        input_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.input_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=10)
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Example button
        example_btn = ttk.Button(input_frame, text="Load Example", command=self.load_example)
        example_btn.pack(anchor=tk.E, padx=5, pady=5)
        
        # Convert button
        convert_btn = ttk.Button(main_frame, text="Convert Script", command=self.convert_script)
        convert_btn.pack(pady=10)
        
        # Output area
        output_frame = ttk.LabelFrame(main_frame, text="Generated Script")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=10)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Action buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Copy to Clipboard", command=self.copy_to_clipboard).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save Script", command=self.save_script).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear All", command=self.clear_all).pack(side=tk.RIGHT, padx=5)
    
    def load_example(self):
        """Load an example script based on the selected language"""
        examples = {
            "Lua": "make part clickable\nmake part unanchored\npart turns red\ncreate GUI for player\nmake GUI green\nwait 1 second\nclose GUI\nend script",
            "Python": "create button\nmake button size 200x50\nset button color blue\nadd text 'Click Me'\nwhen clicked show message\nend script",
            "JavaScript": "create div element\nset width 300px\nset background color yellow\nadd click listener\nshow alert when clicked\nend script",
            "C#": "create form\nadd button to form\nset button text 'Submit'\nmake button clickable\nwhen clicked show message box\nend script",
            "Java": "create JFrame\nset size 400x300\nadd JButton\nset button text 'Press Me'\nadd action listener\nend script"
        }
        
        # More complex examples for AI mode
        ai_examples = {
            "Lua": "Create a part that changes color when clicked\nMake the part float up slowly\nAfter 5 seconds, teleport the part back to original position\nWhen player touches the part, give them 10 points\nCreate a leaderboard showing top scores",
            "Python": "Create a GUI calculator with basic operations\nMake the calculator have memory functions\nAdd a dark mode toggle button\nSave calculation history to a file\nAllow user to load previous calculations",
            "JavaScript": "Create a todo list web app\nAdd ability to mark tasks as complete\nSave tasks to local storage\nAdd drag and drop to reorder tasks\nAdd dark mode support with toggle button",
            "C#": "Create a simple text editor application\nAdd open and save file functionality\nAdd find and replace feature\nImplement undo/redo operations\nAdd word count statistics",
            "Java": "Create a Snake game with arrow key controls\nAdd score display and high score tracking\nMake snake speed increase as score goes up\nAdd pause/resume functionality\nSave high scores to a file"
        }
        
        language = self.lang_var.get()
        mode = self.mode_var.get()
        
        # Choose example based on mode
        if mode == "AI-Powered" and language in ai_examples:
            self.input_text.delete(1.0, tk.END)
            self.input_text.insert(tk.END, ai_examples[language])
        elif language in examples:
            self.input_text.delete(1.0, tk.END)
            self.input_text.insert(tk.END, examples[language])
        else:
            messagebox.showinfo("Info", f"No example available for {language}")
    
    def on_mode_change(self, event=None):
        """Show or hide AI settings based on selected mode"""
        mode = self.mode_var.get()
        
        # Hide both frames first
        self.ai_settings_frame.pack_forget()
        self.online_ai_frame.pack_forget()
        self.local_ai_frame.pack_forget()
        
        if mode == "AI-Powered (Online)":
            self.ai_settings_frame.pack(fill=tk.X, pady=5, after=event.widget.master)
            self.online_ai_frame.pack(fill=tk.X, pady=5)
            if not HAS_OPENAI:
                messagebox.showwarning("Warning", "OpenAI module not found. Please install with:\npip install openai")
        
        elif mode == "AI-Powered (Offline)":
            self.ai_settings_frame.pack(fill=tk.X, pady=5, after=event.widget.master)
            self.local_ai_frame.pack(fill=tk.X, pady=5)
            if not HAS_LOCAL_AI:
                messagebox.showwarning("Warning", "Local AI modules not found. Please install with:\npip install transformers torch")
    
    def setup_local_model(self):
        """Download or setup the selected local model"""
        if not HAS_LOCAL_AI:
            messagebox.showwarning("Warning", "Local AI modules not found. Please install with:\npip install transformers torch")
            return
        
        selected_model = self.local_model_var.get()
        
        # Show loading indicator
        self.model_status_var.set("Status: Setting up model...")
        self.root.update_idletasks()
        
        # Set default model path if custom is not provided
        model_paths = {
            "TinyLlama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "Phi-2": "microsoft/phi-2",
            "GPTQ": "TheBloke/Llama-2-7b-Chat-GPTQ",
            "Custom": self.custom_model_var.get()
        }
        
        # Use a queue to communicate between threads
        result_queue = queue.Queue()
        
        # Process in a separate thread to keep UI responsive
        def download_model():
            try:
                model_path = model_paths.get(selected_model)
                
                if selected_model == "Custom" and not model_path:
                    result_queue.put(("error", "No custom model path provided"))
                    return
                
                # Determine device to use
                device = "cuda" if self.device_var.get() == "CUDA (GPU)" and torch.cuda.is_available() else "cpu"
                
                # Download and load tokenizer (this will download if not cached)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # Setup model with reduced memory usage
                if device == "cuda":
                    # For GPU usage
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        load_in_8bit=True
                    )
                else:
                    # For CPU usage
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )
                
                # Create text generation pipeline
                text_generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=device
                )
                
                # Store as instance variable
                self.local_text_generator = text_generator
                
                result_queue.put(("success", f"Model '{selected_model}' loaded successfully on {device}"))
            except Exception as e:
                result_queue.put(("error", str(e)))
        
        # Start the thread
        threading.Thread(target=download_model, daemon=True).start()
        
        # Check the queue periodically
        def check_queue():
            try:
                status, result = result_queue.get_nowait()
                
                if status == "success":
                    self.model_status_var.set(f"Status: {result}")
                    messagebox.showinfo("Success", result)
                else:
                    self.model_status_var.set(f"Status: Error loading model")
                    messagebox.showerror("Error", f"Model setup failed: {result}")
            except queue.Empty:
                # Queue is empty, check again after a delay
                self.root.after(100, check_queue)
        
        # Start checking the queue
        check_queue()
    
    def test_api_connection(self):
        """Test the OpenAI API connection"""
        api_key = self.api_key_var.get()
        if not api_key:
            messagebox.showwarning("Warning", "Please enter an API key")
            return
            
        if not HAS_OPENAI:
            messagebox.showwarning("Warning", "OpenAI module not found. Please install with:\npip install openai")
            return
            
        try:
            # Set the API key
            openai.api_key = api_key
            
            # Make a simple API call to verify the connection
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.model_var.get(),
                messages=[{"role": "user", "content": "Say 'API connection successful'"}],
                max_tokens=10
            )
            
            messagebox.showinfo("Success", "API connection successful!")
        except Exception as e:
            messagebox.showerror("Error", f"API connection failed: {str(e)}")
    
    def convert_script(self):
        """Convert the input script to the selected language"""
        input_script = self.input_text.get(1.0, tk.END).strip()
        language = self.lang_var.get()
        mode = self.mode_var.get()
        
        if not input_script:
            messagebox.showwarning("Warning", "Please enter some script commands first!")
            return
        
        # Show loading indicator
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Converting script... Please wait.")
        self.root.update_idletasks()
        
        # Use a queue to communicate between threads
        result_queue = queue.Queue()
        
        # Process in a separate thread to keep UI responsive
        def process_script():
            try:
                if mode == "AI-Powered (Online)":
                    result = self.translate_with_online_ai(input_script, language)
                elif mode == "AI-Powered (Offline)":
                    result = self.translate_with_local_ai(input_script, language)
                else:
                    result = self.translate_script(input_script, language)
                result_queue.put(("success", result))
            except Exception as e:
                result_queue.put(("error", str(e)))
        
        # Start the thread
        threading.Thread(target=process_script, daemon=True).start()
        
        # Check the queue periodically
        def check_queue():
            try:
                status, result = result_queue.get_nowait()
                
                # Clear output and display result
                self.output_text.delete(1.0, tk.END)
                
                if status == "success":
                    self.output_text.insert(tk.END, result)
                else:
                    self.output_text.insert(tk.END, f"Error: {result}")
                    messagebox.showerror("Error", f"Conversion failed: {result}")
            except queue.Empty:
                # Queue is empty, check again after a delay
                self.root.after(100, check_queue)
        
        # Start checking the queue
        check_queue()
    
    def translate_with_online_ai(self, script, language):
        """Use online AI (OpenAI) to translate the script to the target language"""
        if not HAS_OPENAI:
            return "Error: OpenAI module not installed. Please install with: pip install openai"
            
        api_key = self.api_key_var.get()
        if not api_key:
            return "Error: OpenAI API key not provided in AI Settings"
            
        try:
            # Set up the OpenAI client
            client = openai.OpenAI(api_key=api_key)
            
            # Prepare the prompt
            prompt = f"""
            Translate the following natural language script commands into {language} code.
            Keep the code clean, efficient, and well-formatted. Add comments where appropriate.
            
            Here are the commands:
            {script}
            
            Please provide valid {language} code that implements these commands.
            """
            
            # Make the API call
            response = client.chat.completions.create(
                model=self.model_var.get(),
                messages=[
                    {"role": "system", "content": f"You are an expert {language} programmer. Your task is to convert natural language commands into valid {language} code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for more deterministic results
                max_tokens=2000
            )
            
            # Extract and return the result
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"AI Translation Error: {str(e)}"
    
    def translate_with_local_ai(self, script, language):
        """Use local AI model to translate the script to the target language"""
        if not HAS_LOCAL_AI:
            return "Error: Local AI modules not installed. Please install with: pip install transformers torch"
        
        try:
            # Check if model is loaded
            if not hasattr(self, 'local_text_generator'):
                return "Error: Local AI model not loaded. Please click 'Download/Setup Model' first."
            
            # Prepare the prompt based on model
            model_name = self.local_model_var.get()
            
            # General prompt template that works with most models
            prompt = f"""
            Task: Convert natural language script commands to {language} programming language.
            
            Commands:
            {script}
            
            {language} Code:
            """
            
            # Generate code using the local model
            max_length = min(len(prompt) + 1500, 2048)  # Limit max length based on model capabilities
            
            generation_params = {
                "max_length": max_length,
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.2,
                "do_sample": True,
                "num_return_sequences": 1
            }
            
            # Generate text
            output = self.local_text_generator(prompt, **generation_params)
            generated_text = output[0]["generated_text"]
            
            # Extract just the code part after the prompt
            code_part = generated_text[len(prompt):].strip()
            
            # If code looks incomplete, add placeholder code
            if len(code_part) < 50:
                code_part += f"\n\n# Note: The local AI model provided limited output.\n# Here's some placeholder {language} code:\n"
                code_part += self.get_placeholder_code(language)
            
            return code_part
            
        except Exception as e:
            return f"Local AI Translation Error: {str(e)}\n\nFalling back to rule-based translation...\n\n" + self.translate_script(script, language)
    
    def get_placeholder_code(self, language):
        """Provide placeholder code when local AI doesn't generate enough"""
        placeholders = {
            "Python": "# Example Python code\nimport time\n\ndef main():\n    print('Hello world')\n    # TODO: Implement functionality\n    time.sleep(1)\n\nif __name__ == '__main__':\n    main()",
            "Lua": "-- Example Lua code\nlocal function main()\n    print('Hello world')\n    -- TODO: Implement functionality\n    wait(1)\nend\n\nmain()",
            "JavaScript": "// Example JavaScript code\nfunction main() {\n    console.log('Hello world');\n    // TODO: Implement functionality\n    setTimeout(() => {\n        console.log('Done');\n    }, 1000);\n}\n\nmain();",
            "C#": "// Example C# code\nusing System;\n\nclass Program {\n    static void Main() {\n        Console.WriteLine(\"Hello world\");\n        // TODO: Implement functionality\n        System.Threading.Thread.Sleep(1000);\n    }\n}",
            "Java": "// Example Java code\npublic class Main {\n    public static void main(String[] args) {\n        System.out.println(\"Hello world\");\n        // TODO: Implement functionality\n        try {\n            Thread.sleep(1000);\n        } catch (InterruptedException e) {}\n    }\n}"
        }
        return placeholders.get(language, "# Placeholder code")
    
    def translate_script(self, script, language):
        """Translate the simple commands to the target language syntax using rule-based approach"""
        lines = script.split('\n')
        result = []
        
        # Maps for different languages
        translations = {
            "Lua": {
                "make part clickable": "part.ClickDetector = Instance.new('ClickDetector', part)",
                "make part unanchored": "part.Anchored = false",
                "part turns red": "part.Color = Color3.fromRGB(255, 0, 0)",
                "create GUI for player": "local gui = Instance.new('ScreenGui', player.PlayerGui)",
                "make GUI green": "gui.BackgroundColor3 = Color3.fromRGB(0, 255, 0)",
                "wait 1 second": "wait(1)",
                "close GUI": "gui:Destroy()",
                "end script": "-- End of script"
            },
            "Python": {
                "create button": "button = tk.Button(root)",
                "make button size 200x50": "button.config(width=200, height=50)",
                "set button color blue": "button.config(bg='blue')",
                "add text 'Click Me'": "button.config(text='Click Me')",
                "when clicked show message": "button.config(command=lambda: messagebox.showinfo('Info', 'Button clicked!'))",
                "end script": "# End of script"
            },
            "JavaScript": {
                "create div element": "const div = document.createElement('div');",
                "set width 300px": "div.style.width = '300px';",
                "set background color yellow": "div.style.backgroundColor = 'yellow';",
                "add click listener": "div.addEventListener('click', function() {",
                "show alert when clicked": "  alert('Div was clicked!');",
                "end script": "});"
            },
            "C#": {
                "create form": "Form form = new Form();",
                "add button to form": "Button button = new Button();",
                "set button text 'Submit'": "button.Text = \"Submit\";",
                "make button clickable": "form.Controls.Add(button);",
                "when clicked show message box": "button.Click += (sender, e) => MessageBox.Show(\"Button clicked!\");",
                "end script": "// End of script"
            },
            "Java": {
                "create JFrame": "JFrame frame = new JFrame();",
                "set size 400x300": "frame.setSize(400, 300);",
                "add JButton": "JButton button = new JButton();",
                "set button text 'Press Me'": "button.setText(\"Press Me\");",
                "add action listener": "button.addActionListener(e -> JOptionPane.showMessageDialog(null, \"Button pressed!\"));",
                "end script": "// End of script"
            }
        }
        
        # Default translator for unknown commands
        def default_translator(cmd, lang):
            # Try to intelligently guess what the command might be doing
            cmd_lower = cmd.lower()
            
            # Common patterns
            if "wait" in cmd_lower or "delay" in cmd_lower or "sleep" in cmd_lower:
                # Extract number if present
                time_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:second|sec|s)', cmd_lower)
                time_val = time_match.group(1) if time_match else "1"
                
                if lang == "Lua":
                    return f"wait({time_val})  -- {cmd}"
                elif lang == "Python":
                    return f"time.sleep({time_val})  # {cmd}"
                elif lang == "JavaScript":
                    return f"setTimeout(() => {{\n  // Continue after delay\n}}, {float(time_val) * 1000});  // {cmd}"
                elif lang == "C#":
                    return f"System.Threading.Thread.Sleep({int(float(time_val) * 1000)});  // {cmd}"
                elif lang == "Java":
                    return f"Thread.sleep({int(float(time_val) * 1000)});  // {cmd}"
            
            elif any(x in cmd_lower for x in ["color", "colour"]):
                color_match = re.search(r'(red|green|blue|yellow|black|white|purple|orange|pink|brown)', cmd_lower)
                color = color_match.group(1) if color_match else "red"
                
                if lang == "Lua":
                    colors = {"red": "255, 0, 0", "green": "0, 255, 0", "blue": "0, 0, 255", 
                             "yellow": "255, 255, 0", "black": "0, 0, 0", "white": "255, 255, 255",
                             "purple": "128, 0, 128", "orange": "255, 165, 0", "pink": "255, 192, 203",
                             "brown": "165, 42, 42"}
                    rgb = colors.get(color, "255, 0, 0")
                    return f"object.Color = Color3.fromRGB({rgb})  -- {cmd}"
                elif lang == "Python":
                    return f"object['bg'] = '{color}'  # {cmd}"
                elif lang == "JavaScript":
                    return f"element.style.backgroundColor = '{color}';  // {cmd}"
                elif lang == "C#":
                    return f"object.BackColor = Color.{color.capitalize()};  // {cmd}"
                elif lang == "Java":
                    return f"object.setBackground(Color.{color.toUpperCase()});  // {cmd}"
            
            # If no pattern is recognized, return a TODO comment
            if lang == "Lua":
                return f"-- TODO: {cmd}"
            elif lang == "Python":
                return f"# TODO: {cmd}"
            elif lang == "JavaScript":
                return f"// TODO: {cmd}"
            elif lang == "C#" or lang == "Java":
                return f"// TODO: {cmd}"
            else:
                return f"-- TODO: {cmd}"
        
        # Add language-specific headers
        if language == "Lua":
            result.append("-- Generated Lua Script")
            result.append("-- Generated on " + datetime.datetime.now().strftime("%Y-%m-%d"))
            result.append("local part = script.Parent")
        elif language == "Python":
            result.append("# Generated Python Script")
            result.append("# Generated on " + datetime.datetime.now().strftime("%Y-%m-%d"))
            result.append("import tkinter as tk")
            result.append("import time")  # Added for sleep functionality
            result.append("from tkinter import messagebox")
            result.append("\nroot = tk.Tk()")
        elif language == "JavaScript":
            result.append("// Generated JavaScript Script")
            result.append("document.addEventListener('DOMContentLoaded', function() {")
        elif language == "C#":
            result.append("// Generated C# Script")
            result.append("using System;")
            result.append("using System.Windows.Forms;")
            result.append("\nnamespace ScriptApp {")
            result.append("    public class Program {")
            result.append("        static void Main() {")
        elif language == "Java":
            result.append("// Generated Java Script")
            result.append("import javax.swing.*;")
            result.append("import java.awt.event.*;")
            result.append("\npublic class GeneratedScript {")
            result.append("    public static void main(String[] args) {")
        
        # Process each line
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            lang_dict = translations.get(language, {})
            translated = lang_dict.get(line, default_translator(line, language))
            result.append(translated)
        
        # Add language-specific footers
        if language == "JavaScript":
            result.append("});")
        elif language == "C#":
            result.append("        }")
            result.append("    }")
            result.append("}")
        elif language == "Java":
            result.append("    }")
            result.append("}")
        
        return "\n".join(result)
    
    def copy_to_clipboard(self):
        """Copy the output text to clipboard"""
        output = self.output_text.get(1.0, tk.END).strip()
        if output:
            self.root.clipboard_clear()
            self.root.clipboard_append(output)
            messagebox.showinfo("Success", "Script copied to clipboard")
        else:
            messagebox.showinfo("Info", "Nothing to copy")
    
    def save_script(self):
        """Save the generated script to a file"""
        output = self.output_text.get(1.0, tk.END).strip()
        if not output:
            messagebox.showinfo("Info", "Nothing to save")
            return
            
        language = self.lang_var.get()
        file_extensions = {
            "Lua": ".lua",
            "Python": ".py",
            "JavaScript": ".js",
            "C#": ".cs",
            "Java": ".java"
        }
        
        ext = file_extensions.get(language, ".txt")
        filename = filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=[
                (f"{language} files", f"*{ext}"),
                ("All files", "*.*")
            ],
            title="Save Script"
        )
        
        if filename:
            with open(filename, 'w') as file:
                file.write(output)
            messagebox.showinfo("Success", f"Script saved as {filename}")
    
    def clear_all(self):
        """Clear both input and output text areas"""
        self.input_text.delete(1.0, tk.END)
        self.output_text.delete(1.0, tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = ScriptMakerApp(root)
    root.mainloop()
