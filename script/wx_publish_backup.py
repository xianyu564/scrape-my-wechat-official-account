# script/wx_publish_backup.py

# wx_publish_backup.py  (year-only folders; HTML named as YYYY-MM-DD_<title>.html)
import os, re, json, time, pathlib, hashlib, html
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ===== 从env.json读取配置 =====
def load_env_config():
    """从env.json文件读取配置"""
    try:
        with open("env.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError("请创建env.json文件，参考env.json.EXAMPLE")
    except json.JSONDecodeError:
        raise ValueError("env.json文件格式错误")

# 加载配置
config = load_env_config()
COOKIE = config.get("COOKIE", "")
TOKEN = config.get("TOKEN", "")  # 发表记录页URL里的 token
WECHAT_ACCOUNT_NAME = config.get("WECHAT_ACCOUNT_NAME", "unknown")

# ===== 可调参数 =====
COUNT_PER_PAGE = int(config.get("COUNT", "20"))   # 建议 20（接口上限）
SLEEP_LIST = float(config.get("SLEEP_LIST", "0.6"))
SLEEP_ART  = float(config.get("SLEEP_ART", "0.3"))
OUTDIR = pathlib.Path(f"backup_{WECHAT_ACCOUNT_NAME}")
TIMEOUT = 30

BASE = "https://mp.weixin.qq.com"
LIST_ENDPOINT = f"{BASE}/cgi-bin/appmsgpublish"

S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Cookie": COOKIE,
    "Referer": "https://mp.weixin.qq.com/"
})

OUTDIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = OUTDIR / "_state.json"
seen = set(json.loads(STATE_FILE.read_text("utf-8")).get("seen", [])) if STATE_FILE.exists() else set()

def save_state():
    STATE_FILE.write_text(json.dumps({"seen": list(seen)}, ensure_ascii=False, indent=2), "utf-8")

def sanitize(name: str) -> str:
    s = re.sub(r"[\\/:*?\"<>|\r\n]+", "_", (name or "untitled")).strip()
    return s[:120] if len(s) > 120 else s

def fetch_publish_page(begin: int, count: int) -> dict:
    """优先走 JSON；失败则从HTML里的 'publish_page = {...};' 抽取"""
    params = {
        "sub": "list",
        "begin": begin,
        "count": count,
        "token": TOKEN,
        "lang": "zh_CN",
        "f": "json",
        "ajax": "1",
    }
    r = S.get(LIST_ENDPOINT, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    # 1) JSON
    try:
        data = r.json()
        if "publish_page" in data:
            return json.loads(data["publish_page"])
    except Exception:
        pass
    # 2) HTML 变量
    html_text = r.text
    m = re.search(r"publish_page\s*=\s*(\{[\s\S]*?\});", html_text)
    if not m:
        raise RuntimeError("未找到 publish_page")
    publish_page_str = html.unescape(m.group(1))
    return json.loads(publish_page_str)

def html_to_md(fragment: str) -> str:
    soup = BeautifulSoup(fragment, "html.parser")
    lines = []
    for node in soup.descendants:
        if not getattr(node, "name", None): continue
        tag = node.name.lower()
        if tag in ("h1","h2","h3"):
            level = {"h1":"#","h2":"##","h3":"###"}[tag]
            lines.append(f"{level} {node.get_text(strip=True)}\n")
        elif tag == "p":
            txt = node.get_text(" ", strip=True)
            if txt: lines.append(txt + "\n")
        elif tag == "img" and node.get("src"):
            alt = node.get("alt") or ""
            lines.append(f"![{alt}]({node.get('src')})\n")
    return "\n".join(lines)

def download(url: str) -> str:
    r = S.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text

def localize_images(content, folder: pathlib.Path):
    imgdir = folder / "images"
    imgdir.mkdir(exist_ok=True)
    soup = BeautifulSoup(content, "html.parser")
    idx = 1
    for img in soup.find_all("img"):
        src = img.get("data-src") or img.get("src")
        if not src or src.startswith("data:"): 
            continue
        full = urljoin(BASE, src)
        try:
            resp = S.get(full, timeout=TIMEOUT)
            if not resp.ok or not resp.content: 
                continue
            ext = ".jpg"
            m = re.search(r"wx_fmt=(\w+)", full)
            if m: ext = "." + m.group(1)
            fn = f"img_{idx:03d}{ext}"
            (imgdir / fn).write_bytes(resp.content)
            img["src"] = f"./images/{fn}"
            if "data-src" in img.attrs: del img["data-src"]
            idx += 1
            time.sleep(0.05)
        except Exception:
            continue
    return str(soup)

def date_str_from_ts(ts: int) -> tuple[str, str]:
    """返回 (YYYY, YYYY-MM-DD)；微信时间戳为 UTC+8，本地用 time.localtime 即可"""
    t = time.localtime(ts or int(time.time()))
    return time.strftime("%Y", t), time.strftime("%Y-%m-%d", t)

def save_article(title: str, link: str, ts: int):
    # 去重
    key = hashlib.md5(link.encode("utf-8")).hexdigest()
    if key in seen:
        return

    html_text = download(link)
    soup = BeautifulSoup(html_text, "html.parser")
    content = soup.select_one("#js_content") or soup.body or soup

    year, ymd = date_str_from_ts(ts)
    safe_title = sanitize(title)
    base_name = f"{ymd}_{safe_title}"              # 用于文件命名
    year_dir = OUTDIR / year                       # 只按年份分目录
    article_dir = year_dir / base_name             # 仍保留每篇一个目录（用于 images）
    article_dir.mkdir(parents=True, exist_ok=True)

    # 图片本地化
    localized = localize_images(str(content), article_dir)

    # 文件名要求：YYYY-MM-DD_<标题>.html
    (article_dir / f"{base_name}.html").write_text(localized, "utf-8")
    # 其它照常：article.md / images / meta.json
    (article_dir / "article.md").write_text(html_to_md(localized), "utf-8")
    meta = {"title": title, "url": link, "publish_time": ts, "year": year, "date": ymd}
    (article_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")

    seen.add(key); save_state()
    print("saved:", (article_dir / f"{base_name}.html").relative_to(OUTDIR))

def extract_links_from_publish_item(item: dict):
    """单次群发里的子文章列表"""
    links = []
    info = item.get("publish_info")
    if not info:
        return links
    if isinstance(info, str):
        info = html.unescape(info)
        try:
            info = json.loads(info)
        except Exception:
            return links
    for a in info.get("appmsgex", []):
        title = a.get("title") or a.get("appmsg_title") or ""
        link  = a.get("link") or a.get("content_url") or a.get("url") or ""
        ts    = a.get("create_time") or a.get("comm_msg_info", {}).get("datetime") or 0
        if link:
            link = urljoin(BASE, link)
            links.append((title, link, ts))
    return links

def main():
    begin = 0
    while True:
        page = fetch_publish_page(begin, COUNT_PER_PAGE)
        publish_list = page.get("publish_list") or []
        if not publish_list:
            break
        for item in publish_list:
            for title, link, ts in extract_links_from_publish_item(item):
                save_article(title, link, ts)
                time.sleep(SLEEP_ART)
        begin += COUNT_PER_PAGE
        time.sleep(SLEEP_LIST)

if __name__ == "__main__":
    if not (COOKIE and TOKEN):
        raise SystemExit("请先设置 WECHAT_BACKEND_COOKIE / WECHAT_BACKEND_TOKEN 环境变量")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    main()
    print("✅ 完成 →", OUTDIR.resolve())
