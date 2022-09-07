from flask import Flask, render_template
import os


def change_order(path, figure_stored_relative_folder_for_vis):
    files = [
        f for f in os.listdir(path) if os.path.isfile(
            os.path.join(
                path,
                f))]
    static_path_files = [
        os.path.join(
            figure_stored_relative_folder_for_vis,
            x) for x in files]
    sorted_static_path_files = []

    suffix_list = [
        "srcip",
        "dstport",
        "proto",
        "pkt",
        "byt",
        "td",
        "srcport",
        "dstip",
        "flow_size",
    ]
    caption_list = ["Source IP popularity rank",
                    "Destination port number distribution",
                    "Protocol distribution",
                    "Number of packets per flow distribution",
                    "Number of bytes per flow distribution",
                    "Time of duration per flow distribution",
                    "Source port number distribution",
                    "Destination IP popularity rank",
                    "Flow size distribution",
                    ]

    for i in range(9):
        suffix = suffix_list[i]
        for _, x in enumerate(static_path_files):
            if suffix in x:
                sorted_static_path_files.append(
                    {"path": x, "caption": caption_list[i]})
                break
    return sorted_static_path_files


class Dashboard():
    def __init__(self, work_folder, figure_stored_relative_folder_for_vis):
        print(work_folder)

        app = Flask(__name__)

        @app.route("/")
        def dashboard():
            path = work_folder

            sorted_static_path_files = change_order(
                path, figure_stored_relative_folder_for_vis)

            return render_template(
                "index.html", static_paths=sorted_static_path_files)

        app.run(debug=True, use_reloader=True, host='0.0.0.0', port=80)
