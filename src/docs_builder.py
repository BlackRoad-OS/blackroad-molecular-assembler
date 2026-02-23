#!/usr/bin/env python3
"""
Documentation site generator for BlackRoad.
Scans markdown files, builds searchable index, generates navigation, and serves locally.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from http.server import HTTPServer, SimpleHTTPRequestHandler
import time
import tempfile
import shutil


@dataclass
class DocPage:
    """Represents a single documentation page."""
    id: str
    title: str
    path: str
    category: str
    content: str
    last_updated: str
    word_count: int


@dataclass
class DocIndex:
    """Search index for documentation."""
    pages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    word_index: Dict[str, List[str]] = field(default_factory=dict)  # word -> [doc_ids]


class DocsBuilder:
    """Builds and manages documentation site."""
    
    VALID_CATEGORIES = {"guides", "api", "tutorials", "reference", "changelog"}
    
    def __init__(self):
        self.pages: Dict[str, DocPage] = {}
        self.index = DocIndex()
    
    def scan_docs(self, root_dir: str) -> List[DocPage]:
        """
        Finds all .md files, parses title from first # heading.
        Returns list of DocPage objects.
        """
        root_path = Path(root_dir)
        if not root_path.exists():
            raise FileNotFoundError(f"Root directory not found: {root_dir}")
        
        pages = []
        for md_file in root_path.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                
                # Extract title from first # heading
                title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
                title = title_match.group(1) if title_match else md_file.stem
                
                # Determine category from directory structure
                rel_path = md_file.relative_to(root_path)
                parts = rel_path.parts
                category = parts[0] if len(parts) > 1 else "uncategorized"
                category = category if category in self.VALID_CATEGORIES else "reference"
                
                # Create document ID
                doc_id = str(rel_path).replace(".md", "").replace("/", "_").replace("-", "_")
                
                # Calculate word count
                word_count = len(content.split())
                
                # Get last modified time
                mtime = os.path.getmtime(md_file)
                last_updated = datetime.fromtimestamp(mtime).isoformat()
                
                page = DocPage(
                    id=doc_id,
                    title=title,
                    path=str(rel_path),
                    category=category,
                    content=content,
                    last_updated=last_updated,
                    word_count=word_count
                )
                pages.append(page)
                self.pages[doc_id] = page
            except Exception as e:
                print(f"Warning: Failed to process {md_file}: {e}", file=sys.stderr)
        
        return pages
    
    def build_index(self, root_dir: str) -> DocIndex:
        """
        Creates searchable index of all docs.
        Scans docs and builds word index for full-text search.
        """
        pages = self.scan_docs(root_dir)
        
        for page in pages:
            # Add page metadata to index
            self.index.pages[page.id] = {
                "title": page.title,
                "path": page.path,
                "category": page.category,
                "last_updated": page.last_updated,
                "word_count": page.word_count
            }
            
            # Build word index (lowercase, alphanumeric only)
            words = set()
            for word in re.findall(r"\b\w+\b", page.content.lower()):
                if len(word) > 2:  # Skip very short words
                    words.add(word)
            
            for word in words:
                if word not in self.index.word_index:
                    self.index.word_index[word] = []
                if page.id not in self.index.word_index[word]:
                    self.index.word_index[word].append(page.id)
        
        return self.index
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Full-text search across all docs.
        Returns list of matching pages with relevance scores.
        """
        query_words = set(re.findall(r"\b\w+\b", query.lower()))
        query_words = {w for w in query_words if len(w) > 2}
        
        if not query_words:
            return []
        
        # Find pages matching any query word
        matching_docs: Dict[str, int] = {}
        for word in query_words:
            if word in self.index.word_index:
                for doc_id in self.index.word_index[word]:
                    matching_docs[doc_id] = matching_docs.get(doc_id, 0) + 1
        
        # Sort by relevance (word matches) and format results
        results = []
        for doc_id, score in sorted(matching_docs.items(), key=lambda x: -x[1]):
            if doc_id in self.index.pages:
                page_info = self.index.pages[doc_id].copy()
                page_info["relevance_score"] = score
                results.append(page_info)
        
        return results
    
    def generate_sidebar(self, root_dir: str) -> Dict[str, Any]:
        """
        Returns nested dict for navigation sidebar.
        Organizes pages by category.
        """
        if not self.pages:
            self.scan_docs(root_dir)
        
        sidebar = {}
        for category in self.VALID_CATEGORIES:
            sidebar[category] = []
        sidebar["other"] = []
        
        for page in sorted(self.pages.values(), key=lambda p: p.title):
            category = page.category if page.category in sidebar else "other"
            sidebar[category].append({
                "id": page.id,
                "title": page.title,
                "path": page.path
            })
        
        # Remove empty categories
        return {k: v for k, v in sidebar.items() if v}
    
    def build_html(self, page_id: str) -> str:
        """
        Converts markdown to basic HTML using stdlib.
        No external dependencies.
        """
        if page_id not in self.pages:
            raise ValueError(f"Page not found: {page_id}")
        
        page = self.pages[page_id]
        markdown = page.content
        html = []
        html.append("<html><head><meta charset='utf-8'><title>{}</title></head><body>".format(
            self._escape_html(page.title)
        ))
        
        # Simple markdown to HTML conversion
        in_code_block = False
        in_list = False
        
        for line in markdown.split("\n"):
            # Code blocks
            if line.strip().startswith("```"):
                if in_code_block:
                    html.append("</pre>")
                    in_code_block = False
                else:
                    html.append("<pre><code>")
                    in_code_block = True
                continue
            
            if in_code_block:
                html.append(self._escape_html(line) + "\n")
                continue
            
            # Headings
            heading_match = re.match(r"^(#+)\s+(.+)$", line)
            if heading_match:
                level = len(heading_match.group(1))
                text = heading_match.group(2)
                html.append(f"<h{level}>{self._escape_html(text)}</h{level}>")
                continue
            
            # Lists
            if line.strip().startswith("- "):
                if not in_list:
                    html.append("<ul>")
                    in_list = True
                item = line.strip()[2:]
                html.append(f"<li>{self._escape_html(item)}</li>")
                continue
            elif in_list:
                html.append("</ul>")
                in_list = False
            
            # Bold and italic
            line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
            line = re.sub(r"\*(.+?)\*", r"<em>\1</em>", line)
            
            # Links
            line = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', line)
            
            # Paragraphs
            if line.strip():
                html.append(f"<p>{line}</p>")
        
        if in_list:
            html.append("</ul>")
        if in_code_block:
            html.append("</pre>")
        
        html.append("</body></html>")
        return "\n".join(html)
    
    def serve(self, port: int = 3000) -> None:
        """
        Simple HTTP server for local preview.
        Serves documentation as static HTML.
        """
        class DocHandler(SimpleHTTPRequestHandler):
            docs_builder = self
            
            def do_GET(self):
                path = self.path.lstrip("/")
                
                if not path or path == "index.html":
                    self._serve_index()
                elif path.endswith(".html"):
                    page_id = path[:-5]
                    self._serve_page(page_id)
                elif path == "sidebar.json":
                    self._serve_sidebar()
                elif path == "search":
                    self._serve_search()
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def _serve_index(self):
                sidebar = self.docs_builder.generate_sidebar(".")
                html = "<html><head><title>Docs</title></head><body><h1>BlackRoad Documentation</h1>"
                for category, pages in sidebar.items():
                    html += f"<h2>{category}</h2><ul>"
                    for page in pages:
                        html += f"<li><a href='/{page['id']}.html'>{page['title']}</a></li>"
                    html += "</ul>"
                html += "</body></html>"
                
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode())
            
            def _serve_page(self, page_id: str):
                try:
                    html = self.docs_builder.build_html(page_id)
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(html.encode())
                except ValueError:
                    self.send_response(404)
                    self.end_headers()
            
            def _serve_sidebar(self):
                sidebar = self.docs_builder.generate_sidebar(".")
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(sidebar).encode())
            
            def _serve_search(self):
                query = self.path.split("?q=")[1] if "?q=" in self.path else ""
                results = self.docs_builder.search(query)
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(results).encode())
            
            def log_message(self, format, *args):
                print(f"[{self.log_date_time_string()}] {format % args}")
        
        server = HTTPServer(("localhost", port), DocHandler)
        print(f"Serving docs on http://localhost:{port}")
        print(f"Press Ctrl+C to stop")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")
    
    def export_json(self, output_path: str) -> None:
        """
        Exports full index as JSON for static site.
        """
        export_data = {
            "pages": self.index.pages,
            "generated_at": datetime.now().isoformat()
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"Exported index to {output_path}")
    
    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters."""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;"))


def main():
    parser = argparse.ArgumentParser(description="BlackRoad documentation builder")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # scan command
    scan_parser = subparsers.add_parser("scan", help="Scan docs directory")
    scan_parser.add_argument("root_dir", help="Root documentation directory")
    
    # build command
    build_parser = subparsers.add_parser("build", help="Build documentation index")
    build_parser.add_argument("root_dir", help="Root documentation directory")
    
    # search command
    search_parser = subparsers.add_parser("search", help="Search documentation")
    search_parser.add_argument("query", help="Search query")
    
    # serve command
    serve_parser = subparsers.add_parser("serve", help="Serve documentation locally")
    serve_parser.add_argument("root_dir", nargs="?", default=".", help="Root documentation directory")
    serve_parser.add_argument("port", nargs="?", default=3000, type=int, help="Port to serve on")
    
    # export command
    export_parser = subparsers.add_parser("export", help="Export index as JSON")
    export_parser.add_argument("root_dir", help="Root documentation directory")
    export_parser.add_argument("output", help="Output JSON file path")
    
    args = parser.parse_args()
    builder = DocsBuilder()
    
    if args.command == "scan":
        pages = builder.scan_docs(args.root_dir)
        print(f"Found {len(pages)} documentation pages")
        for page in pages:
            print(f"  - {page.title} ({page.category})")
    
    elif args.command == "build":
        index = builder.build_index(args.root_dir)
        print(f"Built index with {len(index.pages)} pages")
    
    elif args.command == "search":
        builder.build_index(".")
        results = builder.search(args.query)
        print(f"Found {len(results)} results for '{args.query}'")
        for result in results:
            print(f"  - {result['title']} (score: {result.get('relevance_score', 0)})")
    
    elif args.command == "serve":
        builder.scan_docs(args.root_dir)
        builder.serve(args.port)
    
    elif args.command == "export":
        builder.build_index(args.root_dir)
        builder.export_json(args.output)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
