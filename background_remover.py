import argparse
from rembg import remove
from PIL import Image
import sys

def main():
    parser = argparse.ArgumentParser(description="Remove background from an image.")
    parser.add_argument("input_path", help="Path to the input image.")
    parser.add_argument("output_path", help="Path to save the output image.")
    args = parser.parse_args()

    print(f"Processing image: {args.input_path}")

    try:
        input_image = Image.open(args.input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)

    try:
        output_image = remove(input_image)
    except Exception as e:
        print(f"Error during background removal: {e}")
        # Attempt to provide more specific advice if it's a common rembg/onnxruntime issue
        if " symptômes d'une compilation" in str(e) or "No such file or directory" in str(e) and ".u2net" in str(e):
            print("This might be due to the model file not being downloaded correctly or an issue with onnxruntime.")
            print("Please ensure you have a working internet connection and try running the script again.")
            print("If the problem persists, you might need to clear rembg's model cache or reinstall onnxruntime.")
        sys.exit(1)

    try:
        output_image.save(args.output_path)
        print(f"Output saved to: {args.output_path}")
    except Exception as e:
        print(f"Error saving output image: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
