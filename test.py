from flask import Flask, request

app = Flask(__name__)
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        file.save('C:/Users/豪豪/Envs/torch/mine/classification/1.pdf')
        return '上传成功！'
    else:
        # 渲染上传文件的表单
        return '<form method="POST" enctype="multipart/form-data"><input type="file" name="file"><input type="submit"></form>'


if __name__ == '__main__':
    app.run()
