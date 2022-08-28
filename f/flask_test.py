from flask import Flask, render_template, redirect, url_for, request, flash
import torch
from flask_cors import CORS


app = Flask(__name__)          # 如果是单独应用可以使用__name__，如果是module则用模块名
# Flask还有其他参数https://blog.csdn.net/YZL40514131/article/details/122730037
CORS(app, supports_credentials=True)       # 解决跨域


@app.route('/')
def welcome():                             # 1.自定义路由
    return 'Hello World!'


@app.route('/index', endpoint='shouye')    # 3.默认只响应get请求
def index():
    return render_template('test.html')


@app.route('/redirect')                    # 2.重定向
def redirect_func():   # 直接重定向
    return redirect('https://www.baidu.com')


@app.route('/redirect2')                   # 3.反向路由
def redirect_func2():   # url_for 重定向,跳转到设定了endpoint处
    return redirect(url_for('shouye'))


@app.route('/user/<id>')                   # 4.路由传参1（参数名必须和<>一致）
def show_post(id):                # http://127.0.0.1:5000/user/2
    return f'Post {id}'


@app.route('/users/query_user')            # 5.路由传参2
def hello_getid():                # http://127.0.0.1:5000/users/query_user?id=2
    id = request.args.get('id')
    return 'Hello user!'+id


@app.route('/template/<name>')             # 6.使用模板,并传参
def template(name):               # 会在同级templates文件夹里寻找模板
    return render_template('test2.html', Client_Name=f'{name}')


app.secret_key = '123'


@app.route('/flash/<f>')                   # 7.flash.可以配合条件使用，实现消息提示
def try_flash(f):
    if f == 'f':
        return 'This is flash'
    else:
        flash('这是一个测试')
        flash('出现错误')
        return render_template('test2.html')
    # 渲染的页面通过get_flashed_messages()获取所有的flash值


@app.errorhandler(404)                     # 8.定义错误处理的方法，参数是状态码
def handle_404_error(err):
    """自定义的处理错误方法"""
    # 这个函数的返回值会是前端用户看到的最终结果
    return "出现了404错误， 错误信息：%s" % err


@app.route('/get')                         # 9.get请求
def get_test():
    data = 'test'
    return {'data': data}


@app.route('/post', methods=["POST"])      # 10.post请求
# 获取参数看content-type,见【https://blog.csdn.net/ling620/article/details/107562294】
def post_test():
    print('hello')
    data = 'hello'
    print(request.content_type)
    print(request.json)
    print(request.json.get('clientName'))

    client_name = ''
    ticker = ''
    ric = ''
    size = ''
    price = ''
    currency = ''
    sector = ''
    salesperson = ''
    hp = ''
    flag = ''
    NLP_result = {
        'clientName': client_name,
        'ticker': ticker,
        'ric': ric,
        'size': size,
        'price': price,
        'currency': currency,
        'sector': sector,
        'salesperson': salesperson,
        'hp': hp,
        'flag': flag
    }
    return NLP_result


if __name__ == '__main__':
    app.run()