import os
import yaml
import argparse


def check_data_yaml(yaml_path):
    """
    Check if the data.yaml file exists and has valid paths
    """
    print(f"Checking data.yaml file at: {yaml_path}")

    if not os.path.exists(yaml_path):
        print(f"ERROR: data.yaml file not found at {yaml_path}")
        return False

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        print("\nData YAML contents:")
        print(f"Names: {data.get('names', 'Not found')}")
        print(f"Number of classes: {data.get('nc', 'Not found')}")
        print(f"Train path: {data.get('train', 'Not found')}")
        print(f"Val path: {data.get('val', 'Not found')}")
        print(f"Test path: {data.get('test', 'Not found')}")

        # Check if paths exist
        base_dir = os.path.dirname(yaml_path)

        for key in ['train', 'val', 'test']:
            if key in data:
                path = data[key]
                # Handle absolute and relative paths
                if os.path.isabs(path):
                    full_path = path
                else:
                    # Try relative to yaml file
                    full_path = os.path.join(base_dir, path)
                    # Try relative to current directory
                    if not os.path.exists(full_path):
                        full_path = os.path.join(os.getcwd(), path)

                if os.path.exists(full_path):
                    print(f"{key.capitalize()} path exists: {full_path}")
                else:
                    print(f"WARNING: {key.capitalize()} path does not exist: {full_path}")

                    # Suggest fixes
                    if '../' in path:
                        suggested_path = path.replace('../', 'datasets/')
                        print(f"  Suggestion: Change '{path}' to '{suggested_path}'")

        return True

    except Exception as e:
        print(f"ERROR: Failed to parse data.yaml: {str(e)}")
        return False


def fix_data_yaml(yaml_path):
    """
    Fix common issues in data.yaml file
    """
    if not os.path.exists(yaml_path):
        print(f"ERROR: data.yaml file not found at {yaml_path}")
        return False

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Fix paths
        modified = False
        for key in ['train', 'val', 'test']:
            if key in data and '../' in data[key]:
                old_path = data[key]
                data[key] = old_path.replace('../', 'datasets/')
                print(f"Changed {key} path from '{old_path}' to '{data[key]}'")
                modified = True

        if modified:
            # Backup original file
            backup_path = f"{yaml_path}.bak"
            os.rename(yaml_path, backup_path)
            print(f"Backed up original file to {backup_path}")

            # Write fixed file
            with open(yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)

            print(f"Updated data.yaml file saved to {yaml_path}")
            return True
        else:
            print("No changes needed in data.yaml file")
            return False

    except Exception as e:
        print(f"ERROR: Failed to fix data.yaml: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check and fix data.yaml file')
    parser.add_argument('--path', type=str, default='datasets/data.yaml', help='Path to data.yaml file')
    parser.add_argument('--fix', action='store_true', help='Fix common issues in data.yaml')
    args = parser.parse_args()

    if check_data_yaml(args.path) and args.fix:
        fix_data_yaml(args.path)
