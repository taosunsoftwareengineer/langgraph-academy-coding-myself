import os

def display_image(graph):
    image_data = graph.get_graph().draw_mermaid_png()

    output_path = "graph_output.png"
    with open(output_path, "wb") as f:
        f.write(image_data)

    os.system(f"open {output_path}")