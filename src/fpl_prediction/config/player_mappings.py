"""Manual player name to Understat ID mappings.

These mappings handle cases where FPL and Understat use different names
for the same player (e.g., Brazilian players known by nicknames,
players with special characters, name format differences).
"""

# Manual mapping: FPL name -> Understat player ID
MANUAL_NAME_TO_ID: dict[str, int] = {
    # Brazilian players known by nicknames
    "Carlos Henrique Casimiro": 2248,  # Casemiro
    "Joelinton Cássio Apolinário de Lira": 87,  # Joelinton
    "Norberto Bercique Gomes Betuncal": 9983,  # Beto
    "Richarlison de Andrade": 6026,  # Richarlison
    "Alisson Ramses Becker": 1257,  # Alisson
    "Ederson Santana de Moraes": 6054,  # Ederson
    "Lucas Tolentino Coelho de Lima": 7365,  # Lucas Paquetá
    "Francisco Evanilson de Lima Barbosa": 12963,  # Evanilson
    "Jorge Luiz Frello Filho": 1389,  # Jorginho
    "Willian Borges da Silva": 700,  # Willian
    "Danilo dos Santos de Oliveira": 11317,  # Danilo
    "Gabriel dos Santos Magalhães": 5613,  # Gabriel (Arsenal defender)
    "Gabriel Fernando de Jesus": 5543,  # Gabriel Jesus
    "Gabriel Martinelli Silva": 7752,  # Martinelli
    "Murillo Santiago Costa dos Santos": 12123,  # Murillo
    "Diogo Dalot Teixeira": 7281,  # Diogo Dalot (Man Utd)
    "Diogo Teixeira da Silva": 6854,  # Diogo Jota (Liverpool)
    "Antony Matheus dos Santos": 11094,  # Antony
    "Norberto Murara Neto": 1297,  # Neto (Bournemouth GK)
    "Vini de Souza Costa": 10872,  # Vinicius Souza
    "Felipe Rodrigues da Silva": 7921,  # Felipe (Forest)
    "João Maria Lobo Alves Palhares Costa Palhinha Gonçalves": 10715,  # Palhinha
    # Players with name format differences
    "Benjamin White": 7298,  # Ben White
    "Matty Cash": 8864,  # Matthew Cash
    "Tino Livramento": 9512,  # Valentino Livramento
    "Destiny Udogie": 8831,  # Iyenoma Destiny Udogie
    "Lesley Ugochukwu": 9451,  # Chimuanya Ugochukwu
    "Amad Diallo": 8127,  # Amad Diallo Traore
    "Joe Gomez": 987,  # Joseph Gomez
    "Hwang Hee-chan": 8845,  # Hee-Chan Hwang
    "Yehor Yarmoliuk": 11772,  # Yehor Yarmolyuk
    "João Victor Gomes da Silva": 12766,  # Jota Silva
    "Toti António Gomes": 10293,  # Toti
    "Joe Aribo": 10766,  # Joe Ayodele-Aribo
    "Cheick Doucouré": 8666,  # Cheick Oumar Doucoure
    "Jaden Philogene": 9415,  # Jaden Philogene-Bidace
    "Olu Aina": 725,  # Ola Aina
    "Benoît Badiashile": 7240,  # Benoit Badiashile Mukinayi
    "Alexandre Moreno Lopera": 4120,  # Álex Moreno
    "Maximilian Kilman": 7332,  # Max Kilman
    "Ryan Giles": 7277,  # Ryan John Giles
    "Sam Szmodics": 12752,  # Sammie Szmodics
    "Josh King": 12410,  # Joshua King
    "Nayef Aguerd": 6935,  # Naif Aguerd
    "Ollie Scarles": 12358,  # Oliver Scarles
    "Amari'i Bell": 11713,  # Amari'i Bell (HTML entity in understat)
    "João Pedro Ferreira Silva": 8272,  # Joao Pedro
    "Lewis Cook": 1789,  # Lewis Cook
    "Enes Ünal": 6219,  # Enes Unal
    "Anis Slimane": 11730,  # Anis Ben Slimane
    "Hamed Traorè": 6986,  # Hamed Junior Traore
    "Ben Brereton": 11815,  # Ben Brereton Diaz
    "Carlos Roberto Forbs Borges": 13092,  # Carlos Forbs
    "Arnaud Kalimuendo": 8056,  # Arnaud Kalimuendo Muinga
    "Junior Kroupi": 11504,  # Eli Junior Kroupi
    "Yegor Yarmolyuk": 11772,  # Yehor Yarmolyuk
    "Odysseas Vlachodimos": 375,  # Odisseas Vlachodimos
    "Michale Olakigbe": 11487,  # Michael Olakigbe
    "Will Smallbone": 8224,  # William Smallbone
    "Łukasz Fabiański": 706,  # Lukasz Fabianski
    "Abdukodir Khusanov": 11763,  # Abduqodir Khusanov
    "Nico O'Reilly": 11592,  # Nico O'Reilly (HTML apostrophe)
    "Daniel Ballard": 13715,  # Dan Ballard
    "Luke O'Nien": 14099,  # Luke O'Nien (HTML apostrophe)
    "Trey Nyoni": 12203,  # Treymaurice Nyoni
    "Yéremy Pino Santos": 9024,  # Yeremi Pino
    "Álex Jiménez Sánchez": 12168,  # Alejandro Jimenez
    "Sávio Moreira de Oliveira": 11735,  # Savio (Savinho)
    "Murillo Costa dos Santos": 12123,  # Murillo
    "Jair Paula da Cunha Filho": 13779,  # Jair
    "André Trindade da Costa Neto": 13022,  # Andre
    "Welington Damascena Santos": 13403,  # Welington
    "Estêvão Almeida de Oliveira Gonçalves": 13775,  # Estevao
    "Victor da Silva": 182,  # Victor da Silva
    "Jordan Beyer": 160,  # Jordan Beyer
    "Felipe Augusto de Almeida Monteiro": 445,  # Felipe Augusto
    "Jonathan Castro Otto": 560,  # Jonny Otto
    "Igor Thiago Nascimento Rodrigues": 13222,  # Thiago
    "Francisco Jorge Tomás Oliveira": 10327,  # Francisco Oliveira
    "Kevin Santos Lopes de Macedo": 14030,  # Kevin
    "Rayan Cherki": 8094,  # Mathis Cherki
    "Fer López González": 13200,  # Fernando Lopez
    "Pablo Felipe Pereira de Jesus": 14290,  # Pablo Felipe
    "Mathias Jorgensen": 999999,  # Manual placeholder
    # Players with apostrophes in names
    "Dara O'Shea": 8756,  # Dara O'Shea
    "Jake O'Brien": 12014,  # Jake O'Brien
    "Matt O'Riley": 13206,  # Matt O'Riley
    # Goalkeeper/special cases
    "André Onana": 10913,  # André Onana (different from Amadou Onana)
    "Gabriel Osho": 12151,  # Gabriel Osho
    # Eastern European name differences
    "Đorđe Petrović": 12032,  # Djordje Petrovic
    # Players with nicknames in FPL name
    "Rodrigo 'Rodri' Hernandez": 2496,  # Rodri (Man City)
    "Rodrigo 'Rodri' Hernandez Cascante": 2496,  # Rodri alternate name
    "Rodrigo Hernandez": 2496,  # Rodri without nickname
    "Sávio 'Savinho' Moreira de Oliveira": 11735,  # Savinho (Sávio)
    # Collision fixes (players who share name parts)
    "Jonny Evans": 807,  # Jonny Evans (not Jonny from Wolves)
    "Kyle Walker-Peters": 885,  # Kyle Walker-Peters (not Kyle Walker)
    "Kevin Danso": 5261,  # Kevin Danso
    "Emerson Palmieri dos Santos": 1245,  # Emerson Palmieri (West Ham)
    "Emerson Leite de Souza Junior": 7430,  # Emerson Royal (Spurs)
}

# Common name parts that shouldn't be matched alone (too many collisions)
COMMON_NAME_PARTS: set[str] = {
    "gabriel",
    "andre",
    "rodrigo",
    "lucas",
    "pedro",
    "bruno",
    "matheus",
    "silva",
    "santos",
    "oliveira",
    "souza",
    "lima",
    "costa",
    "ferreira",
    "dos",
    "da",
    "de",
    "do",
    "neto",
    "junior",
    "emerson",
    "kevin",
    "kyle",
    "van",
    "der",
    "den",
    "mohamed",
    "mohammed",
    "jose",
    "carlos",
    "antonio",
}
