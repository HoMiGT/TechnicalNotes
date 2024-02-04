一定要通过vundle来管理vim的插件

安装插件需要在ubuntu系统安装的库

`$ sudo apt install fcitx5 exuberant-ctags vim-nox mono-complete golang nodejs openjdk-17-jdk openjdk-17-jre npm cmake python3-dev`

---

vundle的安装

`$ git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim`

---
`$ vim ~/.vimrc`
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
Plugin 'scrooloose/nerdtree'
Plugin 'fholgado/minibufexpl.vim'
Plugin 'gcmt/wildfire.vim'
Plugin 'sjl/gundo.vim'
Plugin 'Lokaltog/vim-easymotion'
Plugin 'suan/vim-instant-markdown'
Plugin 'lilydjwg/fcitx.vim'
Plug 'neoclide/coc.nvim', {'branch': 'release'}
Plug 'rust-lang/rust.vim' " Rust语法高亮、格式化等
Plug 'prabirshrestha/vim-lsp' " 代码补全、错误检查等
Plug 'scrooloose/nerdtree' " 文件浏览器
Plug 'dense-analysis/ale' " 异步语法检查和代码格式化
Plug 'vim-airline/vim-airline' " 状态栏美化
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
