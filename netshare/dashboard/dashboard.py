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
        "pkt_len",
        "service",
        "conn_state",
        "duration",
        "ts",
        "orig_bytes",
        "resp_bytes",
        "missed_bytes",
        "orig_ip_bytes",
        "resp_ip_bytes"
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
                    "Packet size (bytes)",
                    "Application protocol ID sent over connection",
                    "Connection state",
                    "How long connetion lasted (seconds)",
                    "timestamp of first packet",
                    "Number of payload bytes originator sent",
                    "Number of payload bytes responder sent",
                    "Number of bytes missed (packet loss)",
                    "Number of originator IP bytes",
                    "Number of responder IP bytes"
                    ]

    for i in range(len(suffix_list)):
        suffix = suffix_list[i]
        for _, x in enumerate(static_path_files):
            if suffix in x:
                sorted_static_path_files.append(
                    {"path": x, "caption": caption_list[i]})
                break
    return sorted_static_path_files


class Dashboard():
    def __init__(self, plot_folder, figure_stored_relative_folder_for_vis):
        app = Flask(__name__)

        @app.route("/")
        def dashboard():
            sorted_static_path_files = change_order(
                plot_folder, figure_stored_relative_folder_for_vis)

            return render_template(
                "index.html", static_paths=sorted_static_path_files)

        # Avoid flask typical port 5000 as in MacOS 12 (Monterey)
        # port 5000 is served as the Airplay receiver
        # https://stackoverflow.com/questions/69818376/localhost5000-unavailable-in-macos-v12-monterey
        app.run(debug=True, use_reloader=False, host='localhost', port=8000)
