一定要通过vundle来管理vim的插件

安装插件需要在ubuntu系统安装的库

`$ sudo apt install fcitx5 exuberant-ctags vim-nox mono-complete golang nodejs openjdk-17-jdk openjdk-17-jre npm cmake python3-dev`

---

vundle的安装

`$ git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim`

下载代码补全插件

`git clone --depth=1 --branch release https://gh-proxy.org/https://github.com/neoclide/coc.nvim.git ~/.vim/bundle/coc.nvim`

coc插件需要nodejs 

```
sudo dnf install nodejs
# 要升级node最新版本
node --version
sudo npm install -g n --registry=https://registry.npmmirror.com
# 安装24 长期支持版本
sudo /usr/local/bin/n 24
```     

---
`$ vim ~/.vimrc`
```
" vundle 环境配置
set nocompatible
filetype off
set rtp+=~/.vim/bundle/Vundle.vim
" vundle 插件
call vundle#begin()
Plugin 'VundleVim/Vundle.vim'
Plugin 'altercation/vim-colors-solarized'
Plugin 'scrooloose/nerdtree'
" Plugin 'prabirshrestha/vim-lsp'
Plugin 'OmniSharp/omnisharp-vim'
" Plugin 'Valloric/YouCompleteMe'
Plugin 'neoclide/coc.nvim', {'branch': 'release'}
Plugin 'rust-lang/rust.vim'
Plugin 'vim-airline/vim-airline'
call vundle#end()

syntax enable

filetype plugin indent on 
set showmode
set showcmd
set encoding=utf-8
set t_Co=256
set shiftwidth=4
set textwidth=120
set history=2000
set mouse=a
set tabstop=4
set softtabstop=4
set clipboard+=unnamedplus
set pastetoggle=<F2>


let mapleader=","
" rust.vim 配置
let g:rustfmt_autosave=1
let g:rustfmt_command="rustfmt"

let g:ale_linters = {'rust': ['cargo', 'clippy']}
let g:ale_fixers = {'rust': ['cargo','rustfmt']}

let g:ale_rust_cargo_use_clippy=1
let g:airline#extensions#ale#enabled=1
let g:ycm_server_python_interpreter='/usr/bin/python3'

vnoremap <leader>ft :RustFmtRange<CR>
nnoremap <leader>ft :RustFmt<CR>
nnoremap <M-r> :RustRun<CR>
nnoremap <M-t> :RustTest<CR>

" F5 save and run 
map <F5> :call RunCode()<CR>
imap <F5> <ESC> :call RunCode()<CR>

func! RunCode()
        exec "w"
        if &filetype == 'python'
                exec "!python3 %"
        elseif &filetype =='sh'
                exec "!bash %"
        elseif &filetype =="cpp"
                let compile_cmd = "g++ % -std=c++20 -Wall -o %<"
                let run_cmd = "./%<"
                exec "!" . compile_cmd . " && " .run_cmd 
        endif 
endfunc 

" ============================================================
" =================== 以下为新增的 coc.nvim 配置 ===============
" ============================================================

" 基础系统设置：降低延迟、提升响应速度、不生成备份冲突文件
set nobackup
set nowritebackup
set updatetime=300
set signcolumn=yes

" 1. Tab / Shift-Tab 选择补全候选词
inoremap <silent><expr> <TAB>
      \ coc#pum#visible() ? coc#pum#next(1) :
      \ CheckBackspace() ? "\<Tab>" :
      \ coc#refresh()
inoremap <expr><S-TAB> coc#pum#visible() ? coc#pum#prev(1) : "\<C-h>"

function! CheckBackspace() abort
  let col = col('.') - 1
  return !col || getline('.')[col - 1]  =~# '\s'
endfunction

" 2. 回车键 (<CR>) 确认补全（并自动选中第一项）
inoremap <silent><expr> <CR> coc#pum#visible() ? coc#pum#confirm()
                              \: "\<C-g>u\<CR>\<c-r>=coc#on_enter()\<CR>"

" 3. 跳转导航快捷键
nmap <silent> gd <Plug>(coc-definition)
nmap <silent> gy <Plug>(coc-type-definition)
nmap <silent> gi <Plug>(coc-implementation)
nmap <silent> gr <Plug>(coc-references)

" 4. 诊断信息导航（[g 上一个错误，]g 下一个错误）
nmap <silent> [g <Plug>(coc-diagnostic-prev)
nmap <silent> ]g <Plug>(coc-diagnostic-next)

" 5. 按大写 K 悬停查看函数文档与类型签名
nnoremap <silent> K :call ShowDocumentation()<CR>

function! ShowDocumentation()
  if CocAction('hasProvider', 'hover')
    call CocActionAsync('doHover')
  else
    call feedkeys('K', 'in')
  endif
endfunction

" 6. 代码重命名 (，rn) 与代码快速修复 (，ac)
nmap <leader>rn <Plug>(coc-rename)
nmap <leader>ac <Plug>(coc-codeaction)
```


打开vim配置安装插件

`:PluginInstall`

重新打开vim

`:CocInstall coc-rust-analyzer coc-clangd coc-pyright`



----
以下是旧版本配置

```
" vundle 环境配置
set nocompatible
filetype off
set rtp+=~/.vim/bundle/Vundle.vim
" vundle 管理插件列表必须位于 vundle#begin()和vundle#end()之间
call vundle#begin()
Plugin 'VundleVim/Vundle.vim'
Plugin 'altercation/vim-colors-solarized'
Plugin 'tomasr/molokai'
Plugin 'vim-scripts/phd'
Plugin 'Lokaltog/vim-powerline'
Plugin 'octol/vim-cpp-enhanced-highlight'
Plugin 'nathanaelkane/vim-indent-guides'
Plugin 'derekwyatt/vim-fswitch'
Plugin 'kshenoy/vim-signature'
Plugin 'vim-scripts/BOOKMARKS—Mark-and-Highlight-Full-Lines'
Plugin 'majutsushi/tagbar'
Plugin 'vim-scripts/indexer.tar.gz'
Plugin 'vim-scripts/DfrankUtil'
Plugin 'vim-scripts/vimprj'
Plugin 'dyng/ctrlsf.vim'
Plugin 'terryma/vim-multiple-cursors'
Plugin 'scrooloose/nerdcommenter'
Plugin 'vim-scripts/DrawIt'
Plugin 'SirVer/ultisnips'
Plugin 'Valloric/YouCompleteMe'
Plugin 'derekwyatt/vim-protodef'
Plugin 'fholgado/minibufexpl.vim'
Plugin 'gcmt/wildfire.vim'
Plugin 'sjl/gundo.vim'
Plugin 'Lokaltog/vim-easymotion'
Plugin 'suan/vim-instant-markdown'
Plugin 'lilydjwg/fcitx.vim'
Plugin 'neoclide/coc.nvim', {'branch': 'release'}  " 需要安装node 并且在~/.vim/bundle/coc.nvim 目录下执行 yarn install && yarn build 
Plugin 'rust-lang/rust.vim' " Rust语法高亮、格式化等
Plugin 'prabirshrestha/vim-lsp' " 代码补全、错误检查等
Plugin 'scrooloose/nerdtree' " 文件浏览器
Plugin 'dense-analysis/ale' " 异步语法检查和代码格式化
Plugin 'vim-airline/vim-airline' " 状态栏美化
call vundle#end()
filetype plugin indent on
syntax on
set showmode
set showcmd
set mouse=a
set encoding=utf-8
set t_Co=256
filetype indent on
set autoindent
set tabstop=4
set shiftwidth=4
set number
set relativenumber
set cursorline
set textwidth=120
set nowrap
set scrolloff=5
set sidescrolloff=5
set laststatus=2
set ruler
set showmatch
set hlsearch
set ignorecase
set smartcase
set spell spelllang=en_us
set undofile
set autochdir
set history=2000
set undodir=~/.vim/.undo//
set wildmenu
set wildmode=longest:list,full
" 设置快捷键 在normal模式下 r 就代表要执行之后所有的命令  :! 表示在终端执行  vim内置支持make所以可以不用! 
nnoremap r :!rm -rf main:wa<CR>:make<CR><CR>:cw<CR>:!./main<CR>

let mapleader = ","
" rust.vim 配置
let g:rustfmt_autosave = 1 " 保存时自动格式化
let g:rustfmt_command = "rustfmt"  " 自定义格式化命令

" ALE 配置
let g:ale_linters = {
   'rust': ['cargo', 'clippy'],
}
let g:ale_fixers = {
   'rust': ['cargo', 'rustfmt'],
}
let g:ale_rust_cargo_use_clippy = 1 " 使用 clippy 进行更严格的检查
​
" vim-airline 配置
let g:airline#extensions#ale#enabled = 1 " 在状态栏显示 ALE 检查结果
​
" NERDTree 配置
autocmd vimenter * NERDTree " 编辑文件时默认打开NERDTree
" let g:NERDTreeShowHidden = 1 " 显示隐藏文件
​
" 快捷键映射
" 在可视模式下选中一段文本，执行 ft 进行格式化
vnoremap <leader>ft :RustFmtRange<CR>
" 普通模式下执行 ft 进行格式化
nnoremap <leader>ft :RustFmt<CR>
" 普通模式下，按Meta/Option键+r，执行运行
nnoremap <M-r> :RustRun<CR>
" 普通模式下，按Meta/Option键+t, 执行测试
nnoremap <M-t> :RustTest<CR>
​
" 普通模式下，按 Ctr + t，切换文件列表打开和关闭
nnoremap <C-t> :NERDTreeToggle<CR>
```
--- 

备注：

[参考配置说明](https://www.ruanyifeng.com/blog/2018/09/vimrc.html)
[vim的插件管理文章](https://wizardforcel.gitbooks.io/use-vim-as-ide/content/2.html)
[vim的rust配置文章](https://juejin.cn/post/7250005852710797371)
--- 

配置vim cpp 通过clang提示  
```
$ cd ~/.vim/bundle/YouCompleteMe
$ python3 install.py --clangd-completer
```
