%% ICIC 2026 Conference LaTeX class
%% Reproduces the supplied ICIC 2026 Word template and author formatting rules.
%% Compile with pdfLaTeX. Based on IEEEtran conference mode.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ICIC2026}[2026/08/17 ICIC 2026 Conference Paper Class]

\LoadClass[10pt,conference,a4paper]{IEEEtran}

% Page geometry specified by the conference: A4, top 0.75 in, bottom 1 in,
% left/right 0.63 in. This gives two 3.38 in columns with a 0.25 in gap.
\RequirePackage[a4paper,top=0.75in,bottom=1in,left=0.63in,right=0.63in,
                headsep=0pt,footskip=0.5in]{geometry}
\setlength{\columnsep}{0.25in}

% Times-family fonts. newtx is the standard TeX-compatible Times replacement
% and closely matches Times New Roman metrics without requiring proprietary fonts.
\RequirePackage{newtxtext}
\RequirePackage{newtxmath}

\RequirePackage{graphicx}
\RequirePackage{amsmath}
\RequirePackage{booktabs}
\RequirePackage{cite}
\RequirePackage{url}
\RequirePackage{etoolbox}
\RequirePackage{enumitem}
\RequirePackage{microtype}
\RequirePackage{fancyhdr}

% Conference identifier on the first page only.
\newcommand{\ICICConferenceIdentifier}{979-8-3195-2818-6/26/\$31.00~\textcopyright 2026 IEEE}
\fancypagestyle{ICICfirstpage}{%
  \fancyhf{}%
  \renewcommand{\headrulewidth}{0pt}%
  \renewcommand{\footrulewidth}{0pt}%
  \fancyfoot[L]{\fontsize{10}{12}\selectfont\ICICConferenceIdentifier}%
}

% Base typography and paragraph formatting.
\AtBeginDocument{%
  \fontsize{10}{12}\selectfont
  \setlength{\parindent}{0.20in}%
  \setlength{\parskip}{0pt plus 0pt}% spacing is controlled explicitly below
  \raggedbottom
}

% Title: 24 pt, centered, Times-family.
\makeatletter
\def\@maketitle{%
  \newpage
  \null
  \vskip -0.20in%
  \begin{center}%
    {\fontsize{24}{28}\selectfont\normalfont\@title\par}%
    \vskip 6pt%
    {\@author\par}%
  \end{center}%
  \vskip 4pt%
  \thispagestyle{ICICfirstpage}%
}
\makeatother

% Author blocks: 9 pt. Authors can put 1--4 blocks in a row.
\newcommand{\ICICAuthorBlock}[5]{%
  \begin{minipage}[t]{\linewidth}\centering
    \fontsize{9}{10.8}\selectfont
    #1\\
    \textit{#2}\\
    \textit{#3}\\
    #4\\
    #5
  \end{minipage}%
}
\newcommand{\ICICAuthorRowOne}[1]{%
  \noindent\makebox[\textwidth][c]{\begin{minipage}[t]{0.60\textwidth}\centering #1\end{minipage}}\par\vspace{8pt}}
\newcommand{\ICICAuthorRowTwo}[2]{%
  \noindent\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}p{0.47\textwidth}p{0.47\textwidth}@{}}
    #1 & #2
  \end{tabular*}\par\vspace{8pt}}
\newcommand{\ICICAuthorRowThree}[3]{%
  \noindent\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}p{0.31\textwidth}p{0.31\textwidth}p{0.31\textwidth}@{}}
    #1 & #2 & #3
  \end{tabular*}\par\vspace{8pt}}
\newcommand{\ICICAuthorRowFour}[4]{%
  \noindent\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}p{0.235\textwidth}p{0.235\textwidth}p{0.235\textwidth}p{0.235\textwidth}@{}}
    #1 & #2 & #3 & #4
  \end{tabular*}\par\vspace{8pt}}

% Abstract and keywords: 9 pt, 0.19 in first-line indent.
\renewenvironment{abstract}{%
  \par\vspace{0pt}%
  \begingroup\fontsize{9}{10.8}\selectfont\setlength{\parindent}{0.19in}%
  \noindent\hspace*{0.19in}\textit{Abstract}---\ignorespaces
}{%
  \par\vspace{10pt}\endgroup
}
\renewenvironment{IEEEkeywords}{%
  \par\begingroup\fontsize{9}{10.8}\selectfont\setlength{\parindent}{0.19in}%
  \noindent\hspace*{0.19in}\textit{Keywords}---\ignorespaces
}{%
  \par\vspace{6pt}\endgroup
}

% Main headings: 10 pt small caps, centered, Roman numbering, 8/4 pt spacing.
% Subheadings: 10 pt italic, alphabetic numbering, left aligned, 6/3 pt spacing.
\makeatletter
\renewcommand\section{\@startsection{section}{1}{\z@}%
  {8pt}{4pt}{\normalfont\fontsize{10}{12}\selectfont\centering\scshape}}
\renewcommand\subsection{\@startsection{subsection}{2}{0pt}%
  {6pt}{3pt}{\normalfont\fontsize{10}{12}\selectfont\itshape}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{0.20in}%
  {4pt}{2pt}{\normalfont\fontsize{10}{12}\selectfont\itshape}}
\renewcommand\thesection{\Roman{section}}
\renewcommand\thesubsection{\Alph{subsection}}
\renewcommand\thesubsubsection{\arabic{subsubsection}}
\def\@seccntformat#1{%
  \ifstrequal{#1}{section}{\csname the#1\endcsname.\quad}{%
  \ifstrequal{#1}{subsection}{\csname the#1\endcsname.\quad}{%
  \csname the#1\endcsname)\quad}}}
\makeatother

% Paragraph-after spacing: 6 pt, while retaining 0.2 in first-line indentation.
% IEEEtran normally suppresses paragraph skips, so provide an author-friendly command
% and patch ordinary paragraph breaks through \parskip.
\setlength{\parskip}{6pt}
\setlength{\parindent}{0.20in}

% Equations: 12 pt before and after; equation number remains flush right.
\AtBeginEnvironment{equation}{\vspace{6pt}}
\AfterEndEnvironment{equation}{\vspace{6pt}}
\AtBeginEnvironment{align}{\vspace{6pt}}
\AfterEndEnvironment{align}{\vspace{6pt}}

% Captions. Figures: 8 pt, left aligned and fully justified, below figure,
% 4 pt before / 10 pt after. Tables: 8 pt small caps, centered, above table,
% 12 pt before / 6 pt after.
\makeatletter
\long\def\@makecaption#1#2{%
  \ifx\@captype\@undefined\else
    \ifdefstring{\@captype}{figure}{%
      \vskip 4pt%
      \begingroup\fontsize{8}{9.6}\selectfont\normalfont
        \noindent\parbox{\hsize}{#1.\enspace #2}\par
      \endgroup\vskip 10pt%
    }{%
      \vskip 12pt%
      \begingroup\fontsize{8}{9.6}\selectfont\normalfont\scshape
        \centering #1.\enspace #2\par
      \endgroup\vskip 6pt%
    }%
  \fi
}
\makeatother
\renewcommand{\figurename}{Fig.}
\renewcommand{\tablename}{TABLE}

% Table contents: 8 pt by default when using this environment.
\newenvironment{ICICtable}{\fontsize{8}{9.6}\selectfont}{\normalsize}

% Lists: compact and compatible with column layout.
\setlist[itemize]{leftmargin=0.20in,itemsep=2pt,topsep=2pt,parsep=0pt}
\setlist[enumerate]{leftmargin=0.20in,itemsep=2pt,topsep=2pt,parsep=0pt}

% References: 8 pt IEEE format. The heading is 10 pt small caps, centered.
\makeatletter
\renewenvironment{thebibliography}[1]
 {\section*{REFERENCES}%
  \fontsize{8}{9}\selectfont
  \list{[\arabic{enumiv}]}%
       {\settowidth\labelwidth{[#1]}%
        \leftmargin\labelwidth
        \advance\leftmargin\labelsep
        \usecounter{enumiv}%
        \let\p@enumiv\@empty
        \renewcommand\theenumiv{\arabic{enumiv}}}%
  \sloppy\clubpenalty4000\widowpenalty4000%
  \sfcode`\.=\@m}
 {\def\@noitemerr{\@latex@warning{Empty `thebibliography' environment}}\endlist}
\makeatother

% No page numbers.
\pagestyle{empty}

% Useful unnumbered component heading matching the Word template.
\newcommand{\ICICComponentHeading}[1]{%
  \par\vspace{8pt}{\centering\fontsize{10}{12}\selectfont\scshape #1\par}\vspace{4pt}}

% Funding acknowledgment text on the first page can be supplied with
% \thanks{...} in the title/author metadata, as in IEEE conference papers.

% IEEEtran resets the first-page style at the end of \maketitle; apply the
% conference footer after that reset so it remains visible on page 1 only.
\apptocmd{\maketitle}{\thispagestyle{ICICfirstpage}}{}{}
