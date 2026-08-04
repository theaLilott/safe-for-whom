% ===========================================
% This class was written by Friedrich Günther
% and extended by Alexander Rogovskyy
% ===========================================

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{saartex}[2022/07/11]

\newcounter{intlanguage} %0 = german, 1 = english
\setcounter{intlanguage}{0}
\newcommand{\ifenglish}{}
\newcommand{\ifgerman}{}
\DeclareOption{german}{\setcounter{intlanguage}{0}\renewcommand{\ifenglish}[1]{}\renewcommand{\ifgerman}[1]{#1}}
\DeclareOption{english}{\setcounter{intlanguage}{1}\renewcommand{\ifenglish}[1]{#1}\renewcommand{\ifgerman}[1]{}}
\ProcessOptions\relax

\LoadClass{scrbook} 
\KOMAoptions{DIV=calc, fontsize=12pt, headings=standardclasses, chapterprefix=on, open=right, twoside=true} %DIV=calc produces classical book page layout, open=right makes chapters start on right pages and twoside=true enables left and right pages. It is customary to have the sum of both inner margins be as wide as one outer margin
\RequirePackage{iftex} % We support PdfLaTex and LuaLaTex
\RequirePackage{microtype} % Teaches TeX more involved typographical tricks

\hfuzz5pt % Suppresses warnings of bad boxes up to a size of 5pt
\clubpenalty 50000 % Forces TeX to avoide producing clubs and widows, i.e. first lines of beginning paragraphs on ending pages and last lines of ending paragraphs on new pages 
\widowpenalty 50000

\RequirePackage[toc]{appendix}
\setcounter{tocdepth}{1} % Controls depth of table of contents, 0 displays only parts and chapters, 1 displays parts, chapters and sections, etc.
\RedeclareSectionCommand[tocnumwidth=2em]{chapter} % Controls whitespace boxes set around chapter numbers in TOC

\ifpdftex
	\ifnum\value{intlanguage}=0%
		\RequirePackage[ngerman]{babel} \fi % Correct hyphenation
	\ifnum\value{intlanguage}=1%
		\RequirePackage[english]{babel} \fi % Correct hyphenation
	\RequirePackage[utf8]{inputenc} %Inputencoding for umlauts etc.
	\RequirePackage[T1]{fontenc} %Fontencoding for loading other roman fonts
	\RequirePackage{lmodern} % Latin Modern
	\DeclareFontSeriesDefault[rm]{bf}{b}	
\else
	\RequirePackage{fontspec}
	\ifnum\value{intlanguage}=0%
		\RequirePackage[ngerman]{babel} \fi % Correct hyphenation
	\ifnum\value{intlanguage}=1%
		\RequirePackage[english]{babel} \fi % Correct hyphenation
	\DeclareFontSeriesDefault[rm]{bf}{b}
\fi

\PassOptionsToPackage{backend=biber}{biblatex} % Order of compilation: pdfLaTeX->biber->pdfLaTeX->pdfLaTeX

% Use our custom UdS titlepage
% for alternatives, see 
% https://www.ctan.org/pkg/uni-titlepage
\RequirePackage{uni-titlepage}
\TitlePageStyle{UdS}

\RequirePackage{scrlayer-scrpage}
\pagestyle{scrheadings}
\renewcommand{\headfont}{\small}	
\automark[section]{chapter}
\renewcommand{\thechapter}{\Roman{chapter}} %Redefinition Chapter counter
\renewcommand{\thesection}{\arabic{section}} %Redefinition Section counter

\RequirePackage{enumitem}
\setlist{  
  listparindent=\parindent,
  parsep=0pt,
}

\RequirePackage{dsfont, mathtools, amssymb, delarray, graphicx, nicefrac, import, cancel, mathrsfs, esint, framed, stmaryrd, pdfpages}
\RequirePackage[pdfencoding=unicode, psdextra, colorlinks=false]{hyperref}
\ifnum\value{intlanguage}=0%
	\RequirePackage[style=german]{csquotes} \fi %Correct quotation marks based on locale via \enquote{...}
\ifnum\value{intlanguage}=1%
	\RequirePackage[style=english]{csquotes} \fi 

\renewcommand{\labelenumi}{\(\mathrm{(\roman{enumi})}\)} %Hacky, but works. Produces upright default enumerators in every environment
\renewcommand{\labelenumii}{\(\mathrm{(\arabic{enumii})}\)}
\renewcommand{\labelenumiii}{\(\mathrm{(\alph{enumiii})}\)}

\RequirePackage[amsmath, thmmarks, hyperref]{ntheorem}

% As per cleverref docs, place it after ntheorem but before any specific theorem environments
\RequirePackage[capitalize,nameinlink]{cleveref}

% Predefined Theorems, Lemmata, etc.
\newcommand{\uds@germanlang}{german,ngerman,austrian,naustrian,swissgerman,nswissgerman}
\newcommand{\uds@englishlang}{american,australian,british,canadian,english,newzealand,UKenglish,USenglish}

% Let us first define language dependent labels for the environments
\providecaptionname{\uds@englishlang}{\labelProposition}{Proposition}
\providecaptionname{\uds@germanlang}{\labelProposition}{Proposition}

\providecaptionname{\uds@englishlang}{\labelTheorem}{Theorem}
\providecaptionname{\uds@germanlang}{\labelTheorem}{Satz}

\providecaptionname{\uds@englishlang}{\labelLemma}{Lemma}
\providecaptionname{\uds@germanlang}{\labelLemma}{Lemma}

\providecaptionname{\uds@englishlang}{\labelCorollary}{Corollary}
\providecaptionname{\uds@germanlang}{\labelCorollary}{Korollar}

\theoremstyle{plain}
\theoremheaderfont{\normalfont\bfseries}
\theorembodyfont{\itshape}
\theoremseparator{:}
\theorempreskip{\topsep}
\theorempostskip{\topsep}
\theoremindent0cm
\theoremnumbering{arabic}
\theoremsymbol{}
\newtheorem{Proposition}{\labelProposition}[section] %
\newtheorem{Theorem}[Proposition]{\labelTheorem} %
\newtheorem{Lemma}[Proposition]{\labelLemma} %
\newtheorem{Corollary}[Proposition]{\labelCorollary} %

\providecaptionname{american,australian,british,canadian,english,newzealand,UKenglish,USenglish}{\labelDefinition}{Definition}
\providecaptionname{german,ngerman,austrian,naustrian,swissgerman,nswissgerman}{\labelDefinition}{Definition}

\providecaptionname{american,australian,british,canadian,english,newzealand,UKenglish,USenglish}{\labelReminder}{Reminder}
\providecaptionname{german,ngerman,austrian,naustrian,swissgerman,nswissgerman}{\labelReminder}{Erinnerung}

\providecaptionname{american,australian,british,canadian,english,newzealand,UKenglish,USenglish}{\labelRemark}{Remark}
\providecaptionname{german,ngerman,austrian,naustrian,swissgerman,nswissgerman}{\labelRemark}{Bemerkung}

\providecaptionname{american,australian,british,canadian,english,newzealand,UKenglish,USenglish}{\labelExample}{Example}
\providecaptionname{german,ngerman,austrian,naustrian,swissgerman,nswissgerman}{\labelExample}{Beispiel}

\providecaptionname{american,australian,british,canadian,english,newzealand,UKenglish,USenglish}{\labelMotivation}{Motivation}
\providecaptionname{german,ngerman,austrian,naustrian,swissgerman,nswissgerman}{\labelMotivation}{Motivation}

\providecaptionname{american,australian,british,canadian,english,newzealand,UKenglish,USenglish}{\labelConstruction}{Construction}
\providecaptionname{german,ngerman,austrian,naustrian,swissgerman,nswissgerman}{\labelConstruction}{Konstruktion}

\providecaptionname{american,australian,british,canadian,english,newzealand,UKenglish,USenglish}{\labelNotation}{Notation}
\providecaptionname{german,ngerman,austrian,naustrian,swissgerman,nswissgerman}{\labelNotation}{Notation}

\theoremstyle{plain}
\theoremheaderfont{\normalfont\bfseries}
\theorembodyfont{\upshape}
\theoremseparator{:}
\theorempreskip{\topsep}
\theorempostskip{\topsep}
\theoremindent0cm
\theoremnumbering{arabic}
\theoremsymbol{}
\newtheorem{Definition}[Proposition]{\labelDefinition} %
\newtheorem{Reminder}[Proposition]{\labelReminder} %
\newtheorem{Remark}[Proposition]{\labelRemark} %
\newtheorem{Example}[Proposition]{\labelExample} %
\newtheorem{Motivation}[Proposition]{\labelMotivation} %
\newtheorem{Construction}[Proposition]{\labelConstruction} %
\newtheorem{Notation}[Proposition]{\labelNotation} %

\providecaptionname{american,australian,british,canadian,english,newzealand,UKenglish,USenglish}{\labelProof}{Proof}
\providecaptionname{german,ngerman,austrian,naustrian,swissgerman,nswissgerman}{\labelProof}{Beweis}

\theoremstyle{plain}
\theoremheaderfont{\normalfont\bfseries}
\theorembodyfont{\upshape}
\theoremseparator{:}
\theorempreskip{\topsep}
\theorempostskip{\topsep}
\theoremindent0cm
\theoremnumbering{arabic}
\theoremsymbol{\ensuremath{\square}}
\newtheorem*{Proof}{\labelProof} %

\renewcommand{\theProposition}{\Roman{chapter}.\arabic{section}.\arabic{Proposition}}

%%% Cleveref-Namen für Verweise

\ifnum\value{intlanguage}=0%
	\crefname{Lemma}{Lemma}{Lemmata} %
	\crefname{Definition}{Definition}{Definitionen} %
	\crefname{Example}{Beispiel}{Beispiele} %
	\crefname{Theorem}{Satz}{Sätze} %
	\crefname{Corollary}{Korollar}{Korollare} %
	\crefname{Proposition}{Proposition}{Propositionen} %
	\crefname{Construction}{Konstruktion}{Konstruktionen} %
	\crefname{equation}{Gl.}{Gl.} %
	\crefname{IEEEeqnarray}{Gl.}{Gl.} %
	\crefname{Remark}{Bemerkung}{Bemerkungen} %
	\crefname{Reminder}{Erinnerung}{Erinnerungen} %
	\crefname{figure}{Abbildung}{Abbildungen} %
	\crefname{chapter}{Kapitel}{Kapiteln} %
	\crefname{section}{Abschnitt}{Abschnitte} \fi %
\ifnum\value{intlanguage}=1%
	\crefname{Lemma}{Lemma}{Lemmata} %
	\crefname{Definition}{Definition}{Definitions} %
	\crefname{Example}{Example}{Examples} %
	\crefname{Theorem}{Theorem}{Theorems} %
	\crefname{Corollary}{Corollary}{Corollaries} %
	\crefname{Proposition}{Proposition}{Propositions} %
	\crefname{Construction}{Construction}{Constructions} %
	\crefname{equation}{Eq.}{Eq.} %
	\crefname{IEEEeqnarray}{Eq.}{Eq.} %
	\crefname{Remark}{Remark}{Remarks} %
	\crefname{Reminder}{Reminder}{Reminders} %
	\crefname{figure}{Figure}{Figures} %
	\crefname{chapter}{Chapter}{Chapters} %
	\crefname{section}{Section}{Sections} \fi %

% Quantors with more breathing room
\let\existstemp\exists
\let\foralltemp\forall
\let\nexiststemp\nexists
\newcommand*{\unique}{\mkern3mu\existstemp!\mkern3mu}
\renewcommand*{\exists}{\mkern3mu\existstemp\mkern3mu}
\renewcommand*{\forall}{\mkern3mu\foralltemp\mkern3mu}
\renewcommand*{\nexists}{\mkern3mu\nexiststemp\mkern3mu}
\newcommand{\defeq}{\vcentcolon=}
\newcommand{\eqdef}{=\vcentcolon}

% Real and imaginary parts in upright roman font
\let\Im\relax
\let\Re\relax
\DeclareMathOperator{\Im}{Im}
\DeclareMathOperator{\Re}{Re}
\renewcommand{\imath}{\mathrm{i}\mkern1mu}

% Symbols for disjoint union
\makeatletter
\def\moverlay{\mathpalette\mov@rlay}
\def\mov@rlay#1#2{\leavevmode\vtop{%
   \baselineskip\z@skip \lineskiplimit-\maxdimen
   \ialign{\hfil$\m@th#1##$\hfil\cr#2\crcr}}}
\newcommand{\charfusion}[3][\mathord]{
    #1{\ifx#1\mathop\vphantom{#2}\fi
        \mathpalette\mov@rlay{#2\cr#3}
      }
    \ifx#1\mathop\expandafter\displaylimits\fi}
\makeatother
\newcommand{\cupdot}{\charfusion[\mathbin]{\cup}{\cdot}}
\newcommand{\bigcupdot}{\charfusion[\mathop]{\bigcup}{\cdot}}

% Includes for TikZ
\RequirePackage{tikz}
\usetikzlibrary{cd} 
\usetikzlibrary{babel}

% Various Math operators like ker, det, sin, ... with better spacing
\DeclareMathOperator{\Aut}{Aut}
\DeclareMathOperator{\id}{id}
\DeclareMathOperator{\Gl}{Gl}
\DeclareMathOperator{\cl}{cl}
\DeclareMathOperator{\Int}{Int}
\DeclareMathOperator{\Sym}{Sym}
\DeclareMathOperator{\Hom}{Hom}
\DeclareMathOperator{\Mor}{Mor}
\DeclareMathOperator{\im}{im}
\DeclareMathOperator{\Lin}{Lin}
\DeclareMathOperator{\op}{op}
\DeclareMathOperator{\sgn}{sgn}
\DeclareMathOperator{\ch}{char}
\DeclareMathOperator{\End}{End}
\DeclareMathOperator{\pr}{pr}
\DeclareMathOperator{\dist}{dist}
\DeclareMathOperator{\diag}{diag}
\DeclareMathOperator{\Lie}{Lie}

\RequirePackage{stmaryrd, caption, subcaption}
\SetSymbolFont{stmry}{bold}{U}{stmry}{m}{n}
\RequirePackage{IEEEtrantools} % Provides better version of eqnarray

% Name for Appendix on first appendix page and in TOC
\renewcommand*{\appendixpagename}{Appendix}
\renewcommand*{\appendixtocname}{Appendix}

% Produces typographically correct abbreviations in german. This is an exclusively german feature afaik
\ifnum\value{intlanguage}=0%
	\RequirePackage{xspace} %
	\newcommand{\zB}{\mbox{z.\,B.}\xspace} %
	\newcommand{\zT}{\mbox{z.\,T.}\xspace} %
	\newcommand{\iallg}{\mbox{i.\,allg.}\xspace} %
	\newcommand{\ua}{\mbox{u.\,a.}\xspace} %
	\newcommand{\va}{\mbox{v.\,a.}\xspace} %
	\newcommand{\uU}{\mbox{u.\,U.}\xspace} %
	\newcommand{\idR}{\mbox{i.\,d.\,R.}\xspace} %
	\newcommand{\uae}{\mbox{u.\,ä.}\xspace} %
	\newcommand{\zZ}{\mbox{z.\,Z.}\xspace} %
	\renewcommand{\dh}{\mbox{d.\,h.}\xspace} %
	\newcommand{\usw}{\mbox{usw.}\xspace} %
	\newcommand{\ca}{\mbox{ca.}\xspace} %
	\newcommand{\evtl}{\mbox{evtl.}\xspace} %
	\newcommand{\ff}{\mbox{ff.}\xspace} %
	\newcommand{\sog}{\mbox{sog.}\xspace} %
	\newcommand{\bzw}{\mbox{bzw.}\xspace} %
	\newcommand{\bzgl}{\mbox{bzgl.}\xspace} %
	\newcommand{\dsgl}{\mbox{dsgl.}\xspace} %
	\newcommand{\etc}{\mbox{etc.}\xspace} \fi %