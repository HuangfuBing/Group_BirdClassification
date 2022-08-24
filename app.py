import Using as us
import os
# 导入Flask、render_template、request
from flask import Flask, render_template, request
# 使用类创建一个Flask实例对象app（即创建应用程序）(name 为框架名字)
app = Flask(__name__)
# 开启Flask框架的调试模式
app.config['DEBUG'] = True


# 当前Flask框架的所有业务（写函数来处理浏览器发送过来的请求）
# 路由：通过浏览器访问过来的1请求交给谁来处理
@app.route("/")
def index():
    # 访问制作好的页面
    return render_template("看图识鸟项目前端1.0.html")    # 响应


@app.route("/receive", methods=['POST'])
def receive():
    # 接收图片地址
    image_path = request.form.get("image_path")
    result1, result2 = us.get_result(image_path)
    return render_template("看图识鸟项目前端1.0.html", msg="鸟类俗名："+result1+" & 鸟类学名："+result2)


# 启动应用程序-》启动一个flask项目
if __name__ == '__main__':
    app.run()
