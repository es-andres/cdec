import socketserver
from http.server import BaseHTTPRequestHandler
from urllib import parse
import gensim
from gensim.models import KeyedVectors, FastText
from gensim.test.utils import datapath
import pickle
import sys



# cleaner = pickle.load(open('dwords.p', 'rb'), encoding='latin1')
# print(cleaner['kar'])
# print(len(cleaner))
# sys.exit()


w2v_path = "/home/acrem003/Documents/word_vectors/GoogleNews-vectors-negative300.bin.gz"
fast_text_path = "/home/acrem003/Documents/word_vectors/cc.en.300.bin.gz"
# model = KeyedVectors.load_word2vec_format(datapath(w2v_path), binary=True, limit=1)
model = gensim.models.fasttext.load_facebook_vectors(fast_text_path, encoding='utf-8')


class WVServer(BaseHTTPRequestHandler):

    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def _html(self, message):
        """This just generates an HTML document that includes `message`
        in the body. Override, or re-write this do do more interesting stuff.
        """
        content = f"<html><body><h1>{message}</h1></body></html>"
        return str(message).encode("utf8")  # NOTE: must return a bytes obmemject!

    def do_GET(self):
        q = dict(parse.parse_qsl(parse.urlsplit(self.path).query))
        delim = 'MAKE_LIST'
        p1, p2 = q['p1'], q['p2']
        if p1.startswith(delim):
            p1 = p1.replace(delim, '')
            p1 = p1.split(' ')
        if p2.startswith(delim):
            p2 = p2.replace(delim, '')
            p2 = p2.split(' ')
        res = getattr(model, q['f'])(p1, p2)
        self._set_headers()
        self.wfile.write(self._html(res))


PORT = 8000
try:
    print(f'w2v listening on {PORT}')
    httpd = socketserver.TCPServer(("", PORT), WVServer)
    httpd.serve_forever()
except KeyboardInterrupt:
    httpd.shutdown()
    print('server shut down')
