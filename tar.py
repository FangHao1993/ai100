1、普通tar壓縮命令

tar -zcvf cm-11.tar.gz cm-11

//將cm-11資料夾壓縮成cm-11.tar.gz

 

2、壓縮後的檔案太大，需要將cm-11.tar.gz分割成N個指定大小的檔案，怎麼辦？一條命令搞定

split -b 4000M -d -a 1 cm-11.tar.gz cm-11.tar.gz.

//使用split命令，-b 4000M 表示設定每個分割包的大小，單位還是可以k

// -d "引數指定生成的分割包字尾為數字的形式

//-a x來設定序列的長度(預設值是2)，這裡設定序列的長度為1
3、其實以上兩步也可以合併成一步來執行

tar -zcvf cm-11.tar.gz cm-11 | split -b 4000M -d -a 1 - cm-11.tar.gz.

//採用管道，其中 - 引數表示將所建立的檔案輸出到標準輸出上

 

4、普通解壓命令

tar -zxvf cm-11.tar.gz

 

5、分割後的壓縮包解壓命令如下

cat cm-11.tar.gz.* | tar -zxv

備忘下：

cat a* | tar jxvf -，可以解壓a.tar.bz2.a00類似的。

6、附上tar命令的引數解釋

 

tar可以用來壓縮打包單檔案、多個檔案、單個目錄、多個目錄。

Linux打包命令 tar

tar命令可以用來壓縮打包單檔案、多個檔案、單個目錄、多個目錄。

常用格式：

單個檔案壓縮打包 tar -czvf my.tar.gz file1

多個檔案壓縮打包 tar -czvf my.tar.gz file1 file2,...（file*）（也可以給file*檔案mv 目錄在壓縮）

單個目錄壓縮打包 tar -czvf my.tar.gz dir1

多個目錄壓縮打包 tar -czvf my.tar.gz dir1 dir2

解包至當前目錄：tar -xzvf my.tar.gz

cpio

含子目錄find x* | cpio -o > /y/z.cpio

不含子目錄ls x* | cpio -o > /y/z.cpio

解包： cpio -i < /y/z.cpio

[[email protected] ~]# tar [-cxtzjvfpPN] 檔案與目錄 ....

引數：

-c ：建立一個壓縮檔案的引數指令(create 的意思)；

-x ：解開一個壓縮檔案的引數指令！

-t ：檢視 tarfile 裡面的檔案！

特別注意，在引數的下達中， c/x/t 僅能存在一個！不可同時存在！

因為不可能同時壓縮與解壓縮。

-z ：是否同時具有 gzip 的屬性？亦即是否需要用 gzip 壓縮？

-j ：是否同時具有 bzip2 的屬性？亦即是否需要用 bzip2 壓縮？

-v ：壓縮的過程中顯示檔案！這個常用，但不建議用在背景執行過程！

-f ：使用檔名，請留意，在 f 之後要立即接檔名喔！不要再加引數！

　　　例如使用『 tar -zcvfP tfile sfile』就是錯誤的寫法，要寫成

　　　『 tar -zcvPf tfile sfile』才對喔！

-p ：使用原檔案的原來屬性（屬性不會依據使用者而變）

-P ：可以使用絕對路徑來壓縮！

-N ：比後面接的日期(yyyy/mm/dd)還要新的才會被打包進新建的檔案中！

--exclude FILE：在壓縮的過程中，不要將 FILE 打包！