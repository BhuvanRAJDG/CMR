with open("dashboard.py", encoding="utf-8") as f:
    content = f.read()

content = content.replace("width=\"stretch\"", "use_container_width=True")
content = content.replace("width='stretch'", "use_container_width=True")

with open("dashboard.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done")
