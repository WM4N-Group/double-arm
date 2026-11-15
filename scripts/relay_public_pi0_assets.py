#!/usr/bin/env python3
"""Optional zero-disk byte relay for the official PUBLIC pi0_base objects only.

Bind to localhost and use an SSH reverse forward. This is not an open proxy:
no arbitrary hosts, credentials, URL queries, writes, or filesystem caching.
Weights stream in 256KiB blocks, never stored on this workstation. The remote
receiver still validates sizes and MD5 against the official bucket listing.
"""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import time
from urllib.error import HTTPError
from urllib.parse import unquote
from urllib.request import Request,urlopen


def allowed_path(path):
    decoded=unquote(path)
    return decoded.startswith('/checkpoints/pi0_base/') and '?' not in decoded and '#' not in decoded and '..' not in decoded and '\\' not in decoded


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=18890)
    parser.add_argument('--hours',type=float,default=8)
    args=parser.parse_args();deadline=time.monotonic()+args.hours*3600
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if time.monotonic()>deadline or not allowed_path(self.path):self.send_error(403);return
            headers={}
            if self.headers.get('Range'):headers['Range']=self.headers['Range']
            try:
                with urlopen(Request('https://storage.googleapis.com/openpi-assets'+self.path,headers=headers),timeout=90) as source:
                    self.send_response(source.status)
                    for key in ('Content-Length','Content-Type','Content-Range','Accept-Ranges','ETag'):
                        if source.headers.get(key):self.send_header(key,source.headers[key])
                    self.end_headers()
                    while block:=source.read(256*1024):self.wfile.write(block)
            except HTTPError as error:self.send_error(error.code)
            except (BrokenPipeError,ConnectionResetError,TimeoutError):pass
        def log_message(self,format,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.timeout=2
    print(f'PUBLIC_PI0_RELAY localhost:{args.port} zero-disk, max_hours={args.hours}',flush=True)
    while time.monotonic()<deadline:server.handle_request()
    server.server_close()


if __name__=='__main__':main()
