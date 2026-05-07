from bs4 import BeautifulSoup, NavigableString, Tag, Comment
import re

class HTMLParser:
    """
    A parser to convert HTML content into Markdown, with aggressive noise removal.
    """
    def __init__(self):
        # Tags to remove entirely
        self.noise_tags = [
            "script", "style", "footer", "aside", "form", "iframe", "noscript", 
            "svg", "canvas", "video", "audio", "map", "object", "embed", "applet"
        ]
        # Class/ID patterns for ad/noise containers
        self.noise_attr_pattern = re.compile(
            r'ad-|ads-|tracking|social|popup|cookie|modal|share|related|recommend|promo|outbrain|taboola|disqus|comment', 
            re.I
        )

    def parse(self, html: str) -> str:
        """
        Convert HTML content to a clean Markdown representation.
        """
        if not html:
            return ""
        
        soup = BeautifulSoup(html, "html.parser")
        
        # 1. Pre-process: remove noise tags (guaranteed junk)
        for tag in soup.find_all(self.noise_tags):
            tag.decompose()

        # 1.5. Heuristic-based removal for nav and header
        # These often contain menus but can sometimes contain actual content
        for tag in soup.find_all(["nav", "header"]):
            if self._is_mostly_navigation(tag):
                tag.decompose()

        # 2. Backup the semi-cleaned soup before aggressive filtering.
        # This serves as our fallback if the subsequent cleaning is too aggressive.
        tmp_soup = BeautifulSoup(str(soup), "html.parser")

        # 3. Aggressive noise removal based on classes/IDs
        for attr in ["class", "id"]:
            for element in soup.find_all(attrs={attr: self.noise_attr_pattern}):
                if not isinstance(element, Tag) or element.parent is None:
                    continue
                    
                # Only decompose if it's not a major content tag like article or main
                if element.name in ["article", "main", "body"]:
                    continue
                
                # Safeguard: skip elements that likely contain actual content
                if not element.attrs:
                    continue

                val = element.get(attr, "")
                if isinstance(val, list):
                    val = " ".join(val)
                val = val.lower()
                
                if any(kw in val for kw in ["content", "article", "post", "main", "body", "title"]):
                    continue
                
                element.decompose()

        # 4. Extract content from common content-heavy tags if possible
        content_root = soup.find("article") or soup.body or soup
        markdown = self._process_element(content_root)
        cleaned = self._cleanup(markdown)
        
        if not cleaned.strip():
            # Fallback: if aggressive cleaning removed all content, extract text from the backup state
            fallback_root = tmp_soup.find("article") or tmp_soup.find("main") or tmp_soup.body or tmp_soup
            return fallback_root.get_text(separator="\n", strip=True)
            
        return cleaned

    def _is_mostly_navigation(self, tag: Tag) -> bool:
        """
        Heuristic to determine if a tag (nav/header) is primarily navigation menus.
        """
        all_text = tag.get_text(strip=True)
        if not all_text:
            return True
            
        links = tag.find_all("a")
        if not links:
            return False
            
        links_text = "".join(link.get_text(strip=True) for link in links)
        
        # Link density: proportion of text that is within links
        link_density = len(links_text) / len(all_text)
        
        # Also check for "menu-like" attributes
        classes = tag.get("class", [])
        if not isinstance(classes, list):
            classes = [classes]
        attr_val = " ".join(filter(None, [
            " ".join(classes),
            tag.get("id", "")
        ])).lower()
        
        is_menu_attr = any(kw in attr_val for kw in ["menu", "breadcrumb", "header-nav"])
        
        # Thresholds:
        # <nav> tags are almost always menus unless they have low link density
        # <header> tags often contain titles/metadata, so we are more conservative
        if tag.name == "nav":
            threshold = 0.5 if is_menu_attr else 0.99
        else: # header
            threshold = 0.7 if is_menu_attr else 0.90
            
        # Protect headers that contain a main h1
        if tag.name == "header" and tag.find("h1"):
            return False
        
        return link_density > threshold

    def _process_element(self, element, indent=0) -> str:
        if isinstance(element, Comment):
            return ""
        
        if isinstance(element, NavigableString):
            text = element.string if element.string else ""
            # Quick check for script-like leaks in strings
            if "googletag." in text or "function(" in text or "var " in text:
                return ""
            return text
        
        if not isinstance(element, Tag):
            return ""

        tag_name = element.name.lower()
        
        # Special case for tables
        if tag_name == "table":
            return self._process_table(element)

        # Process children
        children_text = "".join(self._process_element(child, indent + (1 if tag_name in ["ul", "ol"] else 0)) for child in element.children)
        
        if tag_name.lower() in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            level = int(tag_name[1])
            return f"\n\n{'#' * level} {children_text.strip()}\n\n"
        elif tag_name.lower() == "p":
            return f"\n\n{children_text.strip()}\n\n"
        elif tag_name.lower() in ["b", "strong"]:
            text = children_text.strip()
            return f"**{text}**" if text else ""
        elif tag_name.lower() in ["i", "em"]:
            text = children_text.strip()
            return f"*{text}*" if text else ""
        elif tag_name.lower() == "a":
            href = element.get("href", "")
            text = children_text.strip()
            if href and text:
                return f" [{text}]({href}) "
            return text
        elif tag_name.lower() in ["ul", "ol"]:
            return f"\n{children_text}\n"
        elif tag_name == "li":
            return self._process_li(element, children_text, indent)
        elif tag_name.lower() == "code":
            if element.parent and element.parent.name == "pre":
                return children_text
            text = children_text.strip()
            return f" `{text}` " if text else ""
        elif tag_name.lower() == "pre":
            return f"\n```\n{children_text.strip()}\n```\n"
        elif tag_name.lower() == "br":
            return "\n"
        elif tag_name.lower() == "hr":
            return "\n----- \n"
        elif tag_name.lower() in ["div", "section", "article", "main"]:
            # For these, we just wrap them in newlines if they are likely blocks
            return f"\n{children_text}\n"
        else:
            return children_text

    def _process_table(self, element: Tag) -> str:
        rows_data = []
        # Find all rows that belong to this table (not nested ones)
        for row in element.find_all("tr"):
            if row.find_parent("table") == element:
                cells = row.find_all(["td", "th"], recursive=False)
                cols = [self._process_element(td).strip() for td in cells]
                if any(cols): # Only add non-empty rows
                    rows_data.append(cols)
        
        if not rows_data:
            return ""

        # Layout Table Detection: if any cell contains "block" elements, flatten it
        # This is common in older websites like Andrew Barto's
        is_layout_table = False
        for row in rows_data:
            for cell in row:
                if "\n" in cell or "###" in cell or "- " in cell:
                    is_layout_table = True
                    break
            if is_layout_table: break

        if is_layout_table:
            # Flatten layout table: just process cells in order
            content = []
            for row in rows_data:
                for cell in row:
                    if cell:
                        content.append(cell)
            return "\n\n" + "\n\n".join(content) + "\n\n"

        # Otherwise, attempt to build a valid Markdown table
        markdown_rows = []
        max_cols = max(len(row) for row in rows_data) if rows_data else 0
        
        for i, row in enumerate(rows_data):
            # Pad row if necessary
            padded_row = row + [""] * (max_cols - len(row))
            markdown_rows.append("| " + " | ".join(padded_row) + " |")
            
            # Add separator after header row
            if i == 0:
                markdown_rows.append("| " + " | ".join(["---"] * max_cols) + " |")
        
        return "\n\n" + "\n".join(markdown_rows) + "\n\n"

    def _process_li(self, element: Tag, children_text: str, indent: int) -> str:
        prefix = "- "
        if element.parent and element.parent.name == "ol":
            siblings = element.parent.find_all("li", recursive=False)
            try:
                index = siblings.index(element) + 1
                prefix = f"{index}. "
            except ValueError:
                prefix = "1. "
        return f"{'  ' * (indent - 1)}{prefix}{children_text.strip()}\n"

    def _cleanup(self, markdown: str) -> str:
        # 1. Remove common CMS markers and ad-related leaks
        # 2. Remove Google Tag Manager and other script leaks
        markdown = re.sub(r'Google Tag Manager\s*\(noscript\).*?End Google Tag Manager\s*\(noscript\)', '', markdown, flags=re.S | re.I)
        markdown = re.sub(r'googletag\.cmd\.push\(.*\);?', '', markdown)
        markdown = re.sub(r'waitForElementToDisplay\(.*\);?', '', markdown)
        
        # 3. Remove ad unit paths and common tracking text
        markdown = re.sub(r'/\d+/[\w/-]+', '', markdown) # Ad unit paths like /4905536/...
        # 4. Remove leaked HTML tags that BS might have missed
        markdown = re.sub(r'</?(div|script|span|p|br|hr|iframe)[^>]*>', '', markdown, flags=re.I)
        
        # 5. Cleanup whitespace
        markdown = re.sub(r'\r\n', '\n', markdown)
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        
        # Trim each line's trailing whitespace
        lines = [line.rstrip() for line in markdown.split('\n')]
        markdown = '\n'.join(lines)
        
        # Re-collapse newlines that might have been created by stripping
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        
        return markdown.strip()
