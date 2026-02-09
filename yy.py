import os

def sanitize_mlflow_paths(root_dir="mlruns"):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file in ["meta.yaml", "MLmodel"]:
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # This replaces the Windows absolute path with a relative one
                if "file:///C:" in content:
                    print(f"Fixing paths in: {file_path}")
                    # Find the part after 'mlruns' and make it relative
                    new_content = content.split("mlruns")[-1]
                    # Clean up leading slashes and rebuild as relative
                    sanitized = f"artifact_location: ./mlruns{new_content}"
                    
                    # For MLmodel files specifically
                    if file == "MLmodel":
                        # Simplest fix is to set it to an empty string or local path
                        import yaml
                        data = yaml.safe_load(content)
                        data['artifact_path'] = 'model'
                        with open(file_path, 'w') as f:
                            yaml.dump(data, f)
                    else:
                        with open(file_path, 'w') as f:
                            f.write(sanitized)

if __name__ == "__main__":
    sanitize_mlflow_paths()