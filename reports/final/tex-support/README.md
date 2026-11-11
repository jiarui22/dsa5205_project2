# Local LaTeX dependencies

The user-supplied template requires `enumitem` and `titlesec`, which were absent from the local BasicTeX installation. The unmodified package sources were downloaded from CTAN on 2026-11-10 and are loaded only through the project-local `TEXINPUTS` setting. Copyright and licensing notices remain in each source file.

- `enumitem.sty`: https://mirrors.ctan.org/macros/latex/contrib/enumitem/enumitem.sty
- `titlesec.sty`: https://mirrors.ctan.org/macros/latex/contrib/titlesec/titlesec.sty

Other packages are provided by the local TeX distribution. If these packages are already installed, the local support directory is optional.
