#!/usr/bin/env python3
import http.server
import socketserver
import webbrowser
import os
import threading
import time

# Change to the directory containing templates
os.chdir('/Users/mirceaserban/Desktop/Cardio_project_test1/app_code')

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.path = '/templates/index.html'
        return super().do_GET()

PORT = 8080
Handler = MyHTTPRequestHandler

def start_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"✓ Server started at http://localhost:{PORT}")
        print("✓ Opening browser automatically...")
        print("✓ Press Ctrl+C to stop")
        httpd.serve_forever()

def open_browser():
    time.sleep(2)  # Wait for server to start
    webbrowser.open(f'http://localhost:{PORT}')

if __name__ == "__main__":
    # Start browser in background
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start server
    try:
        start_server()
    except KeyboardInterrupt:
        print("\n✓ Server stopped")