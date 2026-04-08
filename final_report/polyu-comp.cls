% polyu-comp.cls
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{polyu-comp}[2025/09/08 PolyU COMP MSc dissertation class]

% Base class
\LoadClass[12pt,a4paper]{article}

% Core packages and global layout
\RequirePackage[a4paper, left=25mm, right=20mm, top=20mm, bottom=20mm]{geometry}
\RequirePackage{graphicx}
\RequirePackage{longtable}
\RequirePackage{lscape}
\RequirePackage{setspace}
\RequirePackage{ragged2e}
\RequirePackage{titlesec}
\RequirePackage{iftex}
\RequirePackage{array} % Added for better table control

% Fonts: pdflatex -> times (near TNR); xelatex/lualatex -> exact Times New Roman
\ifPDFTeX
  \RequirePackage{times}
\else
  \RequirePackage{fontspec}
  \setmainfont{Times New Roman}
\fi

% Global spacing/alignment/pagination
\onehalfspacing
\pagestyle{plain}
\makeatletter
\renewcommand{\ps@plain}{%
  \renewcommand{\@oddhead}{\hfill\thepage}%
  \renewcommand{\@evenhead}{\hfill\thepage}%
  \renewcommand{\@oddfoot}{}%
  \renewcommand{\@evenfoot}{}%
}
\makeatother
\pagestyle{plain}

% Paragraph spacing
\setlength{\parskip}{1em}
\setlength{\parindent}{0pt}

% -------------------- Cover page fields --------------------
\newcommand{\CompTitle}{[PROJECT TITLE]}
\newcommand{\CompStudentName}{[Student Name]}
\newcommand{\CompStudentID}{[Student ID]}
\newcommand{\CompStreamCode}{[Stream Code]} % New
\newcommand{\CompStudentSupervisor}{[Supervisor Name]}
\newcommand{\CompCoExaminer}{[Co-Examiner Name]} % New
\newcommand{\CompAssessor}{[2nd Assessor Name]} % New
\newcommand{\CompSubmissionDate}{[Month Year]} % New

\newcommand{\CompPolyU}{The Hong Kong Polytechnic University \\ Department of Computing}
\newcommand{\CompCode}{COMP4913 Capstone Project}
\newcommand{\CompReportType}{Report (Final)}

% -------------------- Cover Page Layout --------------------
\newcommand{\printCompCover}{%
  \begin{titlepage}
    \thispagestyle{empty}
    \centering

    % 1. University Header
    {\bfseries\fontsize{18pt}{22pt}\selectfont \CompPolyU\par}

    % 2. Course and Report Type
    \vspace{1cm}
    {\fontsize{18pt}{22pt}\selectfont 
     \underline{\CompCode}\par
     \vspace{-0.5cm}
     \CompReportType\par
    }

    % 3. Project Title
    \vspace{2cm}
    {\bfseries\fontsize{24pt}{28pt}\selectfont \CompTitle\par}

    % 4. Student Details Table
    % We use vfill to push this section down, but keep some space at bottom
    \vfill 
    
    % The table is left aligned relative to the page margins
    \begin{flushleft}
    \fontsize{14pt}{18pt}\selectfont
    \renewcommand{\arraystretch}{1.8} % Increases vertical space between rows
    \begin{tabular}{@{}p{6cm} l}
        Student Name: & \CompStudentName \\
        Student ID No.: & \CompStudentID \\
        Programme-Stream Code: & \CompStreamCode \\
        Supervisor: & \CompStudentSupervisor \\
        Co-Examiner: & \CompCoExaminer \\
        2\textsuperscript{nd} Assessor: & \CompAssessor \\
        Submission Date: & \CompSubmissionDate \\
    \end{tabular}
    \end{flushleft}

  \end{titlepage}%
}