# 详细步骤
> 1. 在windows上生成ssh秘钥
>> ssh-keygen -t rsa -f C:\Users\WINDOWS_USER\.ssh\KEY_FILENAME -C USERNAME -b 2048
>> 其中 WINDOWS_USER：是用户名 如 UserA， KEY_FILENAME：是秘钥的文件名，USERNAME: 是google cloud 虚拟机上的用户名
> 2. 将生成的 .pub 公钥文件内的公钥复制到google cloud的元数据里新增的一条 [SSH密钥](https://console.cloud.google.com/compute/metadata)
> 3. 下载 [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)
> 4. 打开 PuTTYgen，点击 File->load private key, 加载秘钥完成后，点击Generate,最后点击 Save private key ，保存私钥为 .ppk 后缀的文件
> 5. 打开 PuTTY
>> Session  
>>    |  
>>    -> Host Name(or IP address) 填写  USERNAME@IP地址  (USERNAME：google cloud 虚拟机上的用户名，同上)  
>> Connection  
>>    |  
>>    SSH  
>>       |  
>>       Auth  
>>          |  
>>          -> Credentials  Private key file for authentication: 选择之前的 .ppk 文件，然后点击open即可  
