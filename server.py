import tornado.ioloop
import tornado.options
import tornado.httpserver
import tornado.autoreload
import tornado.web
import os
import sys
import json
import base64
import matplotlib.pyplot as plt
import pylab
import numpy as np
from PIL import Image
from importlib import reload
from tornado.options import define, options
from recognizer import Recognizer

reload(sys)

img_id = 0
recognizer = Recognizer()


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

    def post(self):
        global img_id
        global recognizer
        b64 = self.get_argument("b64")
        b64 = b64.split(',')[1]
        img_data = base64.b64decode(b64)
        file = open("./temp/" + str(img_id) + ".jpg", "wb")
        file.write(img_data)
        file.close()
        image = Image.open("./temp/" + str(img_id) + ".jpg")
        plt.imshow(image)
        plt.show()
        self.write(json.dumps(recognizer.recognize("./temp/" + str(img_id) + ".jpg")))
        img_id += 1



url = [
    (r'/', IndexHandler)
]

settings = {
    "template_path": os.path.join(os.path.dirname(__file__), "templates"),
    "debug": True,
}

application = tornado.web.Application(
    handlers=url,
    **settings
)

define("port", default=8080, help="run on the given port", type=int)


def main():
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(options.port)
    print("Development server is running at http://10.70.36.65:%s" % options.port)
    print("Quit the server with Control-C")
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    main()
