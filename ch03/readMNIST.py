#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
从MNIST中读取原始图片并保存、读取标签数据并保存。
MNIST文件结构分析可以参考：https://blog.csdn.net/justidle/article/details/103149253
"""
"""
使用方法：
1、将MNIST的文件下载到本地。
2、在py文件所在目录下，建立mnist_data目录。然后将MNIST的四个文件拷贝到mnist_data目录，并解压
3、在py文件所在目录下，建立test目录，改目录用于存放解压出的图片文件和标签文件
"""

import struct
import numpy as np
import PIL.Image
    
def read_image(filename):
    #打开文件
    f = open(filename, 'rb')
    
    #读取文件内容
    index = 0
    buf = f.read()
    
    #关闭文件
    f.close()
    
    #解析文件内容
    #>IIII 表示使用大端规则，读取四个整型
    magic, numImages, rows, columns = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')
    
    for i in range(0, numImages):
        # L代表灰度图片
        image = PIL.Image.new('L', (columns, rows))
        
        for x in range(rows):
            for y in range(columns):
                # ‘>B' 读取一个字节
                image.putpixel((y,x), int(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')
                
        print('save ' + str(i) + 'image')
        image.save('mnist_data/test/'+str(i)+'.png')
        
def read_label(filename, saveFilename):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()
    
    magic, labels = struct.unpack_from('>II' , buf , index)
    index += struct.calcsize('>II')
    
    labelArr = [0] * labels
    
    for x in range(labels):
        labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
        index += struct.calcsize('>B')
    
    save = open(saveFilename, 'w')
    save.write(','.join(map(lambda x: str(x), labelArr)))
    save.write('\n')
    save.close()
    print('save labels success')

if __name__ == '__main__':
    #注意t10k-images-idx3-ubyte里面一共有10,000张图片
    read_image('mnist_data/t10k-images-idx3-ubyte')
    read_label('mnist_data/t10k-labels-idx1-ubyte', 'mnist_data/test/label.txt')
    