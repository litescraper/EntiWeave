from bs4 import BeautifulSoup
from modules.parse.html import HTMLParser
from urllib.parse import urlparse
import re

class WikiParser:
    """
    A specialized parser for wiki-like websites (Wikipedia, Britannica, SEP, etc.).
    Extracts the main article content and cleans up noise while handling links specifically.
    """
    def __init__(self):
        self.html_parser = HTMLParser()
        
        # Generic MediaWiki configuration (common for many wiki sites)
        self.mediawiki_config = {
            "content_selectors": ["#mw-content-text", "div.mw-parser-output"],
            "noise_selectors": [
                "table.metadata", "table.toc", "div.toc",
                "div.hatnote", "div.navbox", "span.mw-editsection", "div.thumb",
                "table.mbox-small", "div.mbox-small", 
                "table.vertical-navbox", ".noprint", ".metadata"
            ],
            "reference_selectors": [
                "div.reflist", "ol.references", ".reference", "sup.reference", 
                "div.mw-references-wrap", ".references", "a[href^='#cite_note']"
            ],
            "internal_link_patterns": ["a.mw-jump-link"]
        }

        # Site-specific overrides or non-MediaWiki configurations
        self.site_configs = {
            "wikipedia.org": self.mediawiki_config,
            "britannica.com": {
                "content_selectors": ["article", ".topic-content", "div.topic-content"],
                "noise_selectors": [
                    ".topic-sidebar", ".topic-media", ".topic-header-tools", 
                    ".topic-paragraph-media", ".related-content", ".ads-container",
                    "aside", ".topic-additional-info", ".topic-footer"
                ],
                "reference_selectors": [
                    ".topic-bibliography", "#bibliography", ".bibliography", 
                    "section#bibliography", ".citation", "sup.cite", "a[href^='#ref']"
                ],
                "internal_link_patterns": []
            },
            "plato.stanford.edu": {
                "content_selectors": ["#main-text"],
                "noise_selectors": ["#auhead", "#nav", "#footer", "#article-sidebar", ".article-sidebar", ".notice"],
                "reference_selectors": [
                    "#bibliography", ".bibliography", "section#bibliography", 
                    "a[href^='#']" # SEP uses internal hashes for citations
                ],
                "internal_link_patterns": []
            },
            "scholarpedia.org": self.mediawiki_config,
            "citizendium.org": self.mediawiki_config,
            "newworldencyclopedia.org": self.mediawiki_config,
            "encyclopediaofmath.org": self.mediawiki_config,
            "wikisource.org": self.mediawiki_config,
            "wikibooks.org": self.mediawiki_config,
            "wikiversity.org": self.mediawiki_config,
            "conservapedia.com": self.mediawiki_config
        }

    def _get_config(self, url: str):
        """Find the matching configuration for the given URL domain."""
        try:
            domain = urlparse(url).netloc.lower()
            for key, config in self.site_configs.items():
                if key in domain:
                    return config
        except Exception:
            pass
        return None

    def is_wiki_site(self, url: str) -> bool:
        """Check if the URL belongs to a known wiki-like site."""
        return self._get_config(url) is not None

    def parse(self, html: str, url: str) -> str:
        """
        Parse the HTML content of a wiki-like site.
        Removes links within the text but preserves them in the reference section.
        """
        if not html:
            return ""

        config = self._get_config(url)
        if not config:
            # Fallback to general HTML parser if no config found
            return self.html_parser.parse(html)

        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # 1. Identify the main content area
            content_root = None
            for sel in config["content_selectors"]:
                content_root = soup.select_one(sel)
                if content_root:
                    break
            
            if not content_root:
                content_root = soup.body or soup

            # 2. Aggressively remove noise sections defined in config
            for sel in config["noise_selectors"]:
                for node in content_root.select(sel):
                    try:
                        node.decompose()
                    except Exception:
                        continue

            # 3. Handle internal jump/utility links
            for pattern in config.get("internal_link_patterns", []):
                for node in content_root.select(pattern):
                    try:
                        node.decompose()
                    except Exception:
                        continue

            # 4. Specifically handle Wikipedia-style infoboxes
            # Convert them to neat Markdown tables before general parsing
            for infobox in content_root.select("table.infobox, div.infobox"):
                try:
                    # If it's a div infobox, it usually contains a table
                    table = infobox if infobox.name == "table" else infobox.find("table")
                    if not table:
                        continue
                    
                    md_rows = []
                    for tr in table.find_all("tr", recursive=True):
                        # Only process rows belonging directly to this infobox
                        if tr.find_parent("table") != table:
                            continue
                        
                        cells = tr.find_all(["th", "td"], recursive=False)
                        if not cells:
                            continue
                        
                        # Get text, preserving some spacing but removing internal garbage
                        cell_texts = [c.get_text(" ", strip=True).replace("|", "\\|") for c in cells]
                        cell_texts = [t for t in cell_texts if t]
                        
                        if not cell_texts:
                            continue
                        
                        if len(cell_texts) == 1:
                            # Full-width header or category
                            md_rows.append(f"| **{cell_texts[0]}** | |")
                        else:
                            # Label and value
                            label = cell_texts[0]
                            value = ", ".join(cell_texts[1:])
                            md_rows.append(f"| {label} | {value} |")
                    
                    if md_rows:
                        # Create a simple table structure
                        infobox_title = "Infobox"
                        # Try to find a header for the infobox title
                        header_cell = table.find(["th", "td"], class_=re.compile(r"fn|infobox-title|infobox-name"))
                        if header_cell:
                            infobox_title = header_cell.get_text(strip=True).replace("|", "\\|")
                        
                        md_table = f"\n\n| {infobox_title} | Details |\n| :--- | :--- |\n" + "\n".join(md_rows) + "\n\n"
                        # Replace the complex HTML with the pre-formatted Markdown
                        infobox.replace_with(soup.new_string(md_table))
                except Exception:
                    continue

            # 5. Handle math formulas specifically to avoid newline noise
            # Wikipedia and other MediaWiki sites often use <span class="mwe-math-element">
            # We want to extract the LaTeX from the alt text of the fallback image if possible
            for math_el in content_root.select(".mwe-math-element, math, .math, .mwe-math-mathml-inline"):
                latex = ""
                # Try to find an image with alt text (common Wikipedia fallback)
                img = math_el.find("img", class_=re.compile(r"mwe-math-fallback"))
                if img and img.get("alt"):
                    latex = img.get("alt")
                else:
                    # Fallback to text content if no img found, but strip it
                    latex = math_el.get_text().strip()
                
                if latex:
                    # Wrap in $ for inline or $$ for block if it's a block element
                    is_block = "display" in math_el.get("style", "") or "block" in math_el.get("class", [])
                    if is_block:
                        math_el.replace_with(f"\n\n$${latex}$$\n\n")
                    else:
                        math_el.replace_with(f" ${latex}$ ")
                else:
                    # If we can't find latex, at least prevent it from being a mess of tags
                    # by replacing it with its stripped text content
                    math_el.replace_with(math_el.get_text().strip())

            # 6. Remove all <a> tags EXCEPT those in designated reference sections
            # First, identify all <a> tags that are inside reference sections
            ref_a_tags = set()
            for sel in config.get("reference_selectors", []):
                for ref_section in content_root.select(sel):
                    # Add all <a> tags inside this reference section
                    ref_a_tags.update(ref_section.find_all("a"))
                    # If the section itself is an <a> tag
                    if ref_section.name == "a":
                        ref_a_tags.add(ref_section)

            # Now iterate over all <a> tags and unwrap those not in reference sections
            for a_tag in content_root.find_all("a"):
                # Preserve if it's in a reference section OR looks like a citation [1], [a], etc.
                text = a_tag.get_text().strip()
                is_citation = bool(re.match(r'^\[(\d+|[a-z])\]$', text, re.I))
                
                if a_tag not in ref_a_tags and not is_citation:
                    try:
                        # unwrap() removes the tag but keeps its inner content
                        a_tag.unwrap()
                    except Exception:
                        continue

            # 7. Use the base HTML parser to convert the cleaned HTML to Markdown
            return self.html_parser.parse(str(content_root))
        except Exception:
            return self.html_parser.parse(html)