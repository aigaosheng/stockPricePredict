from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import base64
import imgkit

def figure2base64(fg):
    img = None
    with BytesIO() as img_buf:
        plt.savefig(img_buf, format='jpg')
        # with Image.open(img_buf) as img:
        #     img.show(title="My Image")

        img = base64.b64encode(img_buf.getbuffer()).decode('utf-8')
    
    return img

def base642image(b64, is_show = False):
    img = Image.open(BytesIO(base64.b64decode(b64)))
    if is_show:
        img.show()
    return img

def image2html(img_lst):
    byml = '<table>\n'
    for lst in img_lst:
        img_tr = []
        for b64 in lst:
            img_ele = '<td>\n<img src="data:image/jpeg;base64,{}" /> \n</td>'.format(b64)
            img_tr.append(img_ele)
        byml += '<tr>\n{}</tr>'.format('\n'.join(img_tr))
    byml += '</table>'
    return byml

def html2image(i_html, o_image_file = None):
    if o_image_file:
        tmp_file = o_image_file
    else:
        tmp_file = 'html_img_temp_12345.jpg'
    imgkit.from_file(i_html, tmp_file)
    print(f'**** save html {i_html} to {tmp_file}')
    with Image.open(tmp_file) as img:
        img_buf = BytesIO()
        img.save(img_buf, format="JPEG")
        img_b64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')    
    return img_b64
