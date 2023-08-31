from flask import Flask, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/fileUpload', methods=['GET','POST'])
def home():
    return 'Hello, World!'

def upload_file():
    if request.method == 'POST':
        f = request.files['file'] # 파일 받기
        d = request.data          # 데이터 받기
        print(f)
        date=d['date']
        time = d['time']
        todo_list_dao.insertimg(f.filename.spilt(".")[0],date,time)
        #저장할 경로 + 파일명
        f.save(".static/"+secure_filename(f.filename))
        return 'uploads'

def insertimg(iname,idate,itime):
    conn = connection.get_connection()
    
    sql = '''
    insert into image(iname, idate, itime) values("{}", "{}", "{}")
    '''
    print(sql)
    
    cursor = conn.cursor()
    ret = cursor.execute(sql.format(iname, idate, itime))
# app.run 안에 debug=True로 명시하면 해당 파일의 코드를 수정할 때마다 Flask가 변경된 것을 인식하고 다시 시작한다.
# 스터디 용도로 코딩을 할 때 내용을 바로 반영해서 결과를 확인하기 편리하다.
if __name__ == '__main__':
    app.run(host = "0.0.0.0", port=8080)