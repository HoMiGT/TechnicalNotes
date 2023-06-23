# vim ~/.vimrc

" vundle 环境配置
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
call vundle#end()
filetype plugin indent on
set nocompatible
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
set wrap
set linebreak
set wrapmargin=2
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


备注：参考配置说明(https://www.ruanyifeng.com/blog/2018/09/vimrc.html)

vim的插件管理文章
https://wizardforcel.gitbooks.io/use-vim-as-ide/content/2.html
