from flask import Flask,request,render_template
from search_api import get_search_result


app = Flask(__name__, template_folder='templates')

@app.route('/')
def search_page():
   return render_template('search_page.html')


@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      query = request.form['query']
      result = get_search_result(query)      
      return render_template("result.html",result = result)


if __name__ == '__main__':
    app.run(host='localhost',port=5000)