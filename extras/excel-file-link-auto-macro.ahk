#Requires AutoHotkey v2.0
#SingleInstance Force

; Excelがアクティブなときだけ有効にする設定
#HotIf WinActive("ahk_exe excel.exe")

; Shift + マウスホイールクリック (Middle Button)
+MButton::
{
    ; 右クリックメニューを表示
    Click "Right"
    Sleep 150  ; メニューが表示されるのを待つ (ミリ秒)

    ; i -> i -> Enter
    Send "i"
    Sleep 50
    Send "i"
    Sleep 50
    Send "{Enter}"
    
    Sleep 300  ; ダイアログ等が出る場合、少し長めに待つ

    ; Tab -> コピー
    Send "{Tab}"
    Sleep 50
    Send "^c"  ; Ctrl+C (コピー)
    Sleep 50

    ; Shift+Tab -> ペースト
    Send "+{Tab}"
    Sleep 50
    Send "^v"  ; Ctrl+V (ペースト)
    Sleep 50

    ; Enter
    Send "{Enter}"
}
#HotIf