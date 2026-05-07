from bs4 import BeautifulSoup
from modules.parse.html import HTMLParser
from urllib.parse import urlparse
import re

class CustomParser:
    """
    A specialized parser for specific high-value websites like GitHub and Reddit.
    Extracts structured data and main content while filtering out noise.
    """
    def __init__(self):
        self.html_parser = HTMLParser()

    def is_custom_site(self, url: str) -> bool:
        """Check if the URL belongs to a supported custom site."""
        try:
            host = urlparse(url).netloc.lower()
            return "github.com" in host or "reddit.com" in host
        except Exception:
            return False

    def parse(self, html: str, url: str) -> str:
        """Parse the HTML content based on the domain."""
        if not html:
            return ""

        try:
            host = urlparse(url).netloc.lower()
            if "github.com" in host:
                return self._parse_github(html, url)
            elif "reddit.com" in host:
                return self._parse_reddit(html, url)
            else:
                return self.html_parser.parse(html)
        except Exception:
            return self.html_parser.parse(html)

    def _parse_github(self, html: str, url: str) -> str:
        """Specific extraction for GitHub repository pages."""
        soup = BeautifulSoup(html, "html.parser")
        
        # 1. Basic Repo Info
        # Targeting specific name elements to avoid notification UI noise
        author_el = soup.select_one("span.author")
        name_el = soup.select_one("strong[itemprop='name']")
        
        if author_el and name_el:
            repo_name = f"{author_el.get_text(strip=True)} / {name_el.get_text(strip=True)}"
        elif name_el:
            repo_name = name_el.get_text(strip=True)
        else:
            # Fallback to page title or URL path
            page_title = soup.select_one("title")
            if page_title and "GitHub" in page_title.get_text():
                repo_name = page_title.get_text().split(":")[0].strip()
            else:
                path_parts = urlparse(url).path.strip("/").split("/")
                if len(path_parts) >= 2:
                    repo_name = f"{path_parts[0]} / {path_parts[1]}"
                else:
                    repo_name = "GitHub Repository"
        
        # 2. About section
        about_section = soup.select_one("div.repository-content div[itemprop='about']") or \
                        soup.select_one(".f4.my-3") or \
                        soup.select_one("p.f4")
        description = about_section.get_text(strip=True) if about_section else "No description available."
        
        # 3. Stats (Stars, Forks, Watching)
        stars = soup.select_one("#repo-stars-counter-star") or \
                soup.select_one(".js-social-count") or \
                soup.select_one("a[href$='/stargazers'] .Counter")
        star_count = stars.get_text(strip=True) if stars else "N/A"
        
        # 4. Topics
        topics = [t.get_text(strip=True) for t in soup.select("a.topic-tag")]
        topics_str = ", ".join(topics) if topics else "None"
        
        # 5. README Content
        readme_container = soup.select_one("#readme") or soup.select_one("article.markdown-body")
        readme_md = self.html_parser.parse(str(readme_container)) if readme_container else "README not found."
        
        # Construct summary
        summary = [
            f"# GitHub Repository: {repo_name}",
            f"\n**URL:** {url}",
            f"**Stars:** {star_count}",
            f"**Topics:** {topics_str}",
            f"\n## About\n{description}",
            f"\n## README\n{readme_md}"
        ]
        
        return "\n".join(summary)

    def _parse_reddit(self, html: str, url: str) -> str:
        """Specific extraction for Reddit thread pages."""
        soup = BeautifulSoup(html, "html.parser")
        
        # Reddit has many UI versions, try to be robust
        # 1. Title
        title_el = soup.select_one("h1") or soup.select_one("[data-ad-id='post-title']")
        title = title_el.get_text(strip=True) if title_el else "Reddit Post"
        
        # 2. Post Content
        # Look for the main post body
        post_content_el = soup.select_one("div[data-click-id='text_content']") or \
                         soup.select_one(".post-content") or \
                         soup.select_one("div[data-test-id='post-content']")
        
        # In newer Reddit, it might be inside shadow DOM or just deeply nested
        if not post_content_el:
            # Fallback: look for the first long-ish text block in an article or main
            main = soup.select_one("main") or soup.select_one("article")
            if main:
                post_content_el = main
        
        post_md = self.html_parser.parse(str(post_content_el)) if post_content_el else ""
        
        # 3. Comments
        comments_section = []
        # Try to find comment elements - handle both old, new, and "newest" Reddit (shreddit)
        comment_els = soup.select("shreddit-comment") or \
                      soup.select("div[data-testid='comment']") or \
                      soup.select(".Comment") or \
                      soup.select("div[id^='t1_']")


        for comment_el in comment_els:
            # Determine depth for indentation (shreddit uses depth attribute)
            depth = 0
            try:
                depth = int(comment_el.get("depth", 0))
            except (ValueError, TypeError):
                # Fallback: check nesting for other Reddit versions
                parent = comment_el.parent
                while parent:
                    if parent.name == "div" and ("Comment" in parent.get("class", []) or parent.get("data-testid") == "comment"):
                        depth += 1
                    parent = parent.parent

            # Extract author - handles shreddit attributes and traditional elements
            author = comment_el.get("author") or ""
            if not author:
                author_el = comment_el.select_one("[data-testid='comment_author']") or \
                            comment_el.select_one(".author") or \
                            comment_el.select_one("[author]")
                if author_el:
                    author = author_el.get("author") or author_el.get_text(strip=True)
            
            if not author:
                author = "Unknown"
            
            body_el = comment_el.select_one("[data-testid='comment_body']") or \
                      comment_el.select_one(".RichTextJSON-root") or \
                      comment_el.select_one("div[data-click-id='text_content']") or \
                      comment_el.select_one("div[slot='comment']") or \
                      comment_el.select_one(".md")
            
            if body_el:
                # Use html_parser to get cleaner markdown for comments too
                comment_text = self.html_parser.parse(str(body_el)).strip()
                if comment_text:
                    indent = "  " * depth
                    # Indent each line of the comment for better hierarchy visualization
                    indented_text = "\n".join([f"{indent}{line}" for line in comment_text.split("\n")])
                    comments_section.append(f"{indent}### Comment by {author}\n{indented_text}\n")
        
        comments_md = "\n".join(comments_section) if comments_section else "No comments found."
        
        # Construct summary
        summary = [
            f"# {title}",
            f"\n**URL:** {url}",
            f"\n## Post Content\n{post_md}",
            f"\n## Discussion\n{comments_md}"
        ]
        
        return "\n".join(summary)

