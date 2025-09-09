
import os
import argparse
import datetime
import json

def get_file_info(filepath):
    """获取文件的创建时间（datetime对象和字符串）和大小。"""
    try:
        stat = os.stat(filepath)
        creation_timestamp = stat.st_ctime
        creation_datetime = datetime.datetime.fromtimestamp(creation_timestamp)
        creation_time_str = creation_datetime.strftime('%Y-%m-%d %H:%M:%S')
        file_size = stat.st_size  # bytes
        return creation_datetime, creation_time_str, file_size
    except OSError:
        return None, None, None

def generate_report(target_dir, json_output_filepath, md_output_filepath):
    """生成文件下载记录报告，包含每篇文章的耗时和大小，以及年度汇总，并保存为JSON和Markdown文件。"""
    articles_data = []

    # 遍历年份目录
    for year_folder in os.listdir(target_dir):
        year_path = os.path.join(target_dir, year_folder)
        if os.path.isdir(year_path) and year_folder.isdigit(): # 确保是目录且是数字年份
            # 遍历文章子文件夹
            for article_folder in os.listdir(year_path):
                article_path = os.path.join(year_path, article_folder)
                if os.path.isdir(article_path):
                    latest_creation_datetime = None
                    article_total_size = 0
                    
                    # 递归遍历文章文件夹内的所有文件
                    for root, _, files in os.walk(article_path):
                        for file in files:
                            filepath = os.path.join(root, file)
                            creation_datetime, _, file_size = get_file_info(filepath)
                            if creation_datetime and file_size is not None:
                                if latest_creation_datetime is None or creation_datetime > latest_creation_datetime:
                                    latest_creation_datetime = creation_datetime
                                article_total_size += file_size
                    
                    if latest_creation_datetime:
                        relative_article_path = os.path.relpath(article_path, target_dir)
                        articles_data.append({
                            'path': relative_article_path,
                            'end_time_datetime': latest_creation_datetime,
                            'size_bytes': article_total_size
                        })

    # 按照文章结束时间排序
    articles_data.sort(key=lambda x: x['end_time_datetime'])

    processed_articles = []
    previous_article_end_time = None
    yearly_summary = {}

    for article in articles_data:
        duration_seconds = 0
        if previous_article_end_time:
            duration_seconds = (article['end_time_datetime'] - previous_article_end_time).total_seconds()
        
        # 提取年份用于年度汇总
        year = article['path'].split(os.sep)[0]
        if year not in yearly_summary:
            yearly_summary[year] = {
                'total_size_bytes': 0,
                'total_download_duration_seconds': 0
            }
        yearly_summary[year]['total_size_bytes'] += article['size_bytes']
        yearly_summary[year]['total_download_duration_seconds'] += duration_seconds

        processed_articles.append({
            'path': article['path'],
            'end_time': article['end_time_datetime'].strftime('%Y-%m-%d %H:%M:%S'),
            'size_bytes': article['size_bytes'],
            'size_mb': round(article['size_bytes'] / (1024 * 1024), 2),
            'duration_from_previous_seconds': round(duration_seconds, 2)
        })
        previous_article_end_time = article['end_time_datetime']
    
    # JSON Output
    report_data_json = {
        "report_generated_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "scan_directory": target_dir,
        "articles": processed_articles,
        "yearly_summary": yearly_summary
    }

    with open(json_output_filepath, 'w', encoding='utf-8') as f:
        json.dump(report_data_json, f, ensure_ascii=False, indent=4)
    print(f"JSON报告已保存到: {json_output_filepath}")

    # Markdown Output
    report_lines_md = ["# 克隆效率报告\n"]
    report_lines_md.append(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines_md.append(f"扫描目录: {target_dir}\n\n")

    report_lines_md.append("## 每篇文章下载详情\n")
    report_lines_md.append("| 文件路径 | 结束时间 | 文件大小 (B) | 文件大小 (MB) | 距上一篇耗时 (s) |\n")
    report_lines_md.append("|---|---|---|---|---|\n")
    for article in processed_articles:
        report_lines_md.append(
            f"| {article['path']} | {article['end_time']} | {article['size_bytes']} | {article['size_mb']:.2f} | {article['duration_from_previous_seconds']:.2f} |\n"
        )
    
    report_lines_md.append("\n## 年度汇总\n")
    report_lines_md.append("| 年度 | 总文件大小 (B) | 总文件大小 (MB) | 总下载耗时 (s) |\n")
    report_lines_md.append("|---|---|---|---|\n")
    # Sort yearly summary by year
    sorted_yearly_summary = sorted(yearly_summary.items())
    for year, data in sorted_yearly_summary:
        total_size_mb = round(data['total_size_bytes'] / (1024 * 1024), 2)
        report_lines_md.append(
            f"| {year} | {data['total_size_bytes']} | {total_size_mb:.2f} | {data['total_download_duration_seconds']:.2f} |\n"
        )

    with open(md_output_filepath, 'w', encoding='utf-8') as f:
        f.writelines(report_lines_md)
    print(f"Markdown报告已保存到: {md_output_filepath}")

def main():
    parser = argparse.ArgumentParser(description='生成指定目录下文件的下载记录报告。')
    parser.add_argument('--target_dir', type=str, default='Wechat-Backup/文不加点的张衔瑜', help='要扫描的目标目录。')
    args = parser.parse_args()

    base_output_dir = os.path.join(os.getcwd(), args.target_dir)
    
    if not os.path.exists(base_output_dir):
        print(f"错误: 目标目录不存在: {base_output_dir}")
        return

    json_output_filepath = os.path.join(base_output_dir, '克隆效率.json')
    md_output_filepath = os.path.join(base_output_dir, '克隆效率.md')
    
    generate_report(args.target_dir, json_output_filepath, md_output_filepath)

if __name__ == '__main__':
    # 示例调用：直接传入目标目录
    # 你也可以在命令行运行：python script/generate_clone_report.py --target_dir "Wechat-Backup/文不加点的张衔瑜"
    class MockArgs:
        def __init__(self, target_dir):
            self.target_dir = target_dir
    
    # 使用你的目标文件夹路径
    mock_args = MockArgs('..\Wechat-Backup\文不加点的张衔瑜') 

    base_output_dir = os.path.join(os.getcwd(), mock_args.target_dir)
    
    if not os.path.exists(base_output_dir):
        print(f"错误: 目标目录不存在: {base_output_dir}")
    else:
        json_output_filepath = os.path.join(base_output_dir, '克隆效率.json')
        md_output_filepath = os.path.join(base_output_dir, '克隆效率.md')
        
        generate_report(mock_args.target_dir, json_output_filepath, md_output_filepath)
