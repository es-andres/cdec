import socketserver
from http.server import BaseHTTPRequestHandler
from urllib import parse
import gensim
import shared_vars as vars

model = gensim.models.fasttext.load_facebook_vectors(vars.WORD_VECS_PATH, encoding='utf-8')


class WVServer(BaseHTTPRequestHandler):

    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

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
        self.wfile.write(str(res).encode("utf8"))


PORT = 8000
try:
    print(f'w2v listening on {PORT}')
    httpd = socketserver.TCPServer(("", PORT), WVServer)
    httpd.serve_forever()
except KeyboardInterrupt:
    httpd.shutdown()
    print('server shut down')
