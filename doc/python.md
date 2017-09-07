# python知识点详细

### 1.切片
<pre><code>
print str[0:3]      #截取第一位到第三位的字符
print str[:]        #截取字符串的全部字符
print str[6:]       #截取第七个字符到结尾
print str[:-3]      #截取从头开始到倒数第三个字符之前
print str[2]        #截取第三个字符
print str[-1]       #截取倒数第一个字符
print str[::-1]     #创造一个与原字符串顺序相反的字符串
print str[-3:-1]    #截取倒数第三位与倒数第一位之前的字符
print str[-3:]      #截取倒数第三位到结尾
print str[:-5:-3]   #逆序截取   
</code></pre>
>列表切片产生了副本，不与原列表共用内存
<pre><code>
x=[1,2,3]
y=x[:]
x[0]=-1
print(y)            #输出[1,2,3]
</code></pre>