from PIL import Image

def visualize_results(query_image_path, result_image_paths):
    query_image = Image.open(query_image_path)
    query_image.show(title="Query Image")
    for result_path in result_image_paths:
        result_image = Image.open(result_path)
        result_image.show(title="Result Image")
