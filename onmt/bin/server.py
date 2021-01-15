#!/usr/bin/env python
import configargparse

from flask import Flask, jsonify, request
from waitress import serve
from onmt.translate import TranslationServer, ServerModelError
import logging
from logging.handlers import RotatingFileHandler
from onmt.utils.parse import ArgumentParser

STATUS_OK = "ok"
STATUS_ERROR = "error"


def start(
    config_file,
    url_root="./translator",
    host="0.0.0.0",
    port=5000,
    debug=False,
):
    def prefix_route(route_function, prefix="", mask="{0}{1}"):
        def newroute(route, *args, **kwargs):
            return route_function(mask.format(prefix, route), *args, **kwargs)

        return newroute

    if debug:
        logger = logging.getLogger("main")
        log_format = logging.Formatter(
            "[%(asctime)s %(levelname)s] %(message)s"
        )
        file_handler = RotatingFileHandler(
            "debug_requests.log", maxBytes=1000000, backupCount=10
        )
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    app = Flask(__name__)
    app.route = prefix_route(app.route, url_root)
    translation_server = TranslationServer()
    translation_server.start(config_file)

    @app.route("/models", methods=["GET"])
    def get_models():
        out = translation_server.list_models()
        return jsonify(out)

    @app.route("/health", methods=["GET"])
    def health():
        out = {}
        out["status"] = STATUS_OK
        return jsonify(out)

    @app.route("/clone_model/<int:model_id>", methods=["POST"])
    def clone_model(model_id):
        out = {}
        data = request.get_json(force=True)
        timeout = -1
        if "timeout" in data:
            timeout = data["timeout"]
            del data["timeout"]

        opt = data.get("opt", None)
        try:
            model_id, load_time = translation_server.clone_model(
                model_id, opt, timeout
            )
        except ServerModelError as e:
            out["status"] = STATUS_ERROR
            out["error"] = str(e)
        else:
            out["status"] = STATUS_OK
            out["model_id"] = model_id
            out["load_time"] = load_time

        return jsonify(out)

    @app.route("/unload_model/<int:model_id>", methods=["GET"])
    def unload_model(model_id):
        out = {"model_id": model_id}

        try:
            translation_server.unload_model(model_id)
            out["status"] = STATUS_OK
        except Exception as e:
            out["status"] = STATUS_ERROR
            out["error"] = str(e)

        return jsonify(out)

    def update_server_model_opt(server_model, new_opt):
        # see translation_server l369 to check opts and parse
        # parser = ArgumentParser()
        # print(parser._action_groups[0].title)
        # print(server_model.opt)
        # parse_args = ""
        # print(parser._action_groups)
        # print(parser._actions)
        
        # for (k, v) in old_translate_opt.items():
        #     if k not in new_opt_dict:
        #         parse_args += f"-{k} {v} "
        # for (k, v) in new_opt_dict.items():
        #     parse_args += f"-{k} {v} "
        # new_opt = parser.parse_args(parse_args)
        # ArgumentParser.validate_translate_opts(new_opt)
        server_model.translator.update_translate_opt(*new_opt)
        # maybe does not work:
        server_model.opt.update(new_opt)
        ArgumentParser.validate_translate_opts(new_opt)


    @app.route("/huggingface_proxy", methods=["POST"])
    def huggingface_proxy():
        data = request.get_json(force=True)
        src = data["context"]
        # translation_server.models[100].opt.n_best = 6
        # translation_server.models[100].translator.n_best = 6
        data["opt"] = ""
        update_server_model_opt(translation_server.models[100], data["opt"])
        out = translate_from_inputs([{"src": src, "id": 100}])
        sentences = []
        for trans in out:
            sentences.append(
                {
                    "value": trans[0]["tgt"],
                    "time": 1,
                    "token": len(trans[0]["tgt"].split()),
                }
            )
        return jsonify({"sentences": sentences})

    def translate_from_inputs(inputs):
        if debug:
            logger.info(inputs)
        out = {}
        try:
            trans, scores, n_best, _, aligns = translation_server.run(inputs)
            assert len(trans) == len(inputs) * n_best
            assert len(scores) == len(inputs) * n_best
            assert len(aligns) == len(inputs) * n_best

            out = [[] for _ in range(n_best)]
            for i in range(len(trans)):
                response = {
                    "src": inputs[i // n_best]["src"],
                    "tgt": trans[i],
                    "n_best": n_best,
                    "pred_score": scores[i],
                }
                if len(aligns[i]) > 0 and aligns[i][0] is not None:
                    response["align"] = aligns[i]
                out[i % n_best].append(response)
        except ServerModelError as e:
            model_id = inputs[0].get("id")
            if debug:
                logger.warning(
                    "Unload model #{} " "because of an error".format(model_id)
                )
            translation_server.models[model_id].unload()
            out["error"] = str(e)
            out["status"] = STATUS_ERROR
        if debug:
            logger.info(out)
        return out

    @app.route("/translate", methods=["POST"])
    def translate():
        inputs = request.get_json(force=True)
        out = translate_from_inputs(inputs)
        return jsonify(out)

    @app.route("/to_cpu/<int:model_id>", methods=["GET"])
    def to_cpu(model_id):
        out = {"model_id": model_id}
        translation_server.models[model_id].to_cpu()

        out["status"] = STATUS_OK
        return jsonify(out)

    @app.route("/to_gpu/<int:model_id>", methods=["GET"])
    def to_gpu(model_id):
        out = {"model_id": model_id}
        translation_server.models[model_id].to_gpu()

        out["status"] = STATUS_OK
        return jsonify(out)

    serve(app, host=host, port=port)


def _get_parser():
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        description="OpenNMT-py REST Server",
    )
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default="5000")
    parser.add_argument("--url_root", type=str, default="/translator")
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument(
        "--config", "-c", type=str, default="./available_models/conf.json"
    )
    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()
    start(
        args.config,
        url_root=args.url_root,
        host=args.ip,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
