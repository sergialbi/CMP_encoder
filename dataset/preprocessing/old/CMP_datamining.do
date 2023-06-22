import excel "cmp_filtered_manifestos_Spain.xlsx", clear first

destring date party_code cmp_code , replace

bysort id: gen id_manifesto= _n

gen party_name=party_code
label define party_name 33020 "Unidas Podemos" 33025 "Podemos" 33091 "Geroa Bai" 33092 "Amaiur" 33093 "Compromís" 33095 "EH Bildu" 33097 "En marea" 33210 "Podemos" 33220 "IU" 33230 "Más País" 33320 "PSOE" 33420 "C's" 33440 "UPyD" 33610 "PP" 33611 "CiU" 33612 "FAC" 33710 "Vox" 33902 "PNV/EAJ" 33903 "EA" 33906 "PA" 33907 "CC-PNC" 33909 "CHA" 33910 "UPN" 33913 "PRC" 33916 "Teruel Existe"
label values party_name party_name

replace language= "galician" if language=="spanish" & party_code==33097

forvalues i = 1/9 {
    drop if text=="`i'."
	drop if text=="`i')"
}

replace text = subinstr(text, "••", " ", .)

replace text = subinstr(text, "  ", " ", .)

foreach l in – - �  • > {
    replace text = subinstr(text, "`l'", "", .)
}

foreach l in A B C D E F G H I J K L M N O P Q R S T U V W X Y Z {
   replace text = subinstr(text, "o`l'", "`l'", .)
}

drop if text==""

export excel C:\Users\mguinjoanc\Desktop\cmp_filtered_manifestos_Spain_revised.xlsx, firstrow(var) replace

