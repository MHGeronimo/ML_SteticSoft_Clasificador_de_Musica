# utils.py (Versión con listas extensas de canciones y artistas)

GENRE_DATA = {
    "blues": {
        "songs": [
            "B.B. King - The Thrill Is Gone",
            "Muddy Waters - Mannish Boy",
            "John Lee Hooker - Boom Boom",
            "Howlin' Wolf - Spoonful",
            "Robert Johnson - Cross Road Blues",
            "Etta James - I'd Rather Go Blind",
            "Albert King - Born Under a Bad Sign",
            "Stevie Ray Vaughan - Pride and Joy",
            "Buddy Guy - Damn Right, I've Got the Blues",
            "Lead Belly - Where Did You Sleep Last Night"
        ],
        "artists": [
            "B.B. King", "Muddy Waters", "John Lee Hooker", "Howlin' Wolf", "Robert Johnson",
            "Etta James", "Albert King", "Buddy Guy", "Lead Belly", "Bessie Smith",
            "Stevie Ray Vaughan", "T-Bone Walker", "Willie Dixon", "Son House", "Lightnin' Hopkins",
            "Elmore James", "Big Bill Broonzy", "Memphis Minnie", "Koko Taylor", "Freddie King",
            "J. B. Lenoir", "Otis Rush", "Magic Sam", "Skip James", "Big Mama Thornton",
            "Sonny Boy Williamson II", "Jimmy Reed", "Clarence 'Gatemouth' Brown", "Professor Longhair", "Albert Collins"
        ]
    },
    "classical": {
        "songs": [
            "Ludwig van Beethoven - Symphony No. 5",
            "Wolfgang Amadeus Mozart - Eine kleine Nachtmusik",
            "Johann Sebastian Bach - Cello Suite No. 1",
            "Antonio Vivaldi - The Four Seasons: Spring",
            "Pyotr Ilyich Tchaikovsky - Swan Lake Suite",
            "Claude Debussy - Clair de Lune",
            "George Frideric Handel - Messiah: Hallelujah Chorus",
            "Frédéric Chopin - Nocturne in E-flat Major, Op. 9 No. 2",
            "Igor Stravinsky - The Rite of Spring",
            "Richard Wagner - Ride of the Valkyries"
        ],
        "artists": [
            "Ludwig van Beethoven", "Wolfgang Amadeus Mozart", "Johann Sebastian Bach", "Pyotr Ilyich Tchaikovsky", "Frédéric Chopin",
            "Antonio Vivaldi", "Igor Stravinsky", "Claude Debussy", "George Frideric Handel", "Franz Schubert",
            "Richard Wagner", "Johannes Brahms", "Joseph Haydn", "Gustav Mahler", "Sergei Rachmaninoff",
            "Giuseppe Verdi", "Erik Satie", "Hector Berlioz", "Franz Liszt", "Dmitri Shostakovich",
            "Aaron Copland", "Leonard Bernstein", "Philip Glass", "John Williams", "Ennio Morricone",
            "Hildegard von Bingen", "Clara Schumann", "Fanny Mendelssohn", "Arcangelo Corelli", "Antonín Dvořák"
        ]
    },
    "country": {
        "songs": [
            "Johnny Cash - I Walk the Line",
            "Dolly Parton - Jolene",
            "Willie Nelson - On the Road Again",
            "Hank Williams - I'm So Lonesome I Could Cry",
            "Patsy Cline - Crazy",
            "Garth Brooks - Friends in Low Places",
            "George Strait - Amarillo by Morning",
            "Shania Twain - Man! I Feel Like a Woman!",
            "Kenny Rogers - The Gambler",
            "John Denver - Take Me Home, Country Roads"
        ],
        "artists": [
            "Johnny Cash", "Hank Williams", "Dolly Parton", "Willie Nelson", "Patsy Cline",
            "George Strait", "Garth Brooks", "Merle Haggard", "Loretta Lynn", "Waylon Jennings",
            "Shania Twain", "Kenny Rogers", "Tammy Wynette", "Alan Jackson", "Reba McEntire",
            "George Jones", "John Denver", "Kris Kristofferson", "Vince Gill", "Trisha Yearwood",
            "Brad Paisley", "Carrie Underwood", "Blake Shelton", "Tim McGraw", "Faith Hill",
            "The Chicks (Dixie Chicks)", "Jimmie Rodgers", "Carter Family", "Buck Owens", "Glen Campbell"
        ]
    },
    "disco": {
        "songs": [
            "Bee Gees - Stayin' Alive",
            "Donna Summer - I Feel Love",
            "Earth, Wind & Fire - September",
            "ABBA - Dancing Queen",
            "Chic - Le Freak",
            "Gloria Gaynor - I Will Survive",
            "The Trammps - Disco Inferno",
            "Kool & The Gang - Celebration",
        ],
        "artists": [
            "Donna Summer", "Bee Gees", "Earth, Wind & Fire", "Chic", "ABBA",
            "KC and the Sunshine Band", "Kool & The Gang", "Gloria Gaynor", "The Trammps", "Sister Sledge",
            "Village People", "Diana Ross", "Barry White", "Sylvester", "Giorgio Moroder",
            "Boney M.", "Lipps Inc.", "A Taste of Honey", "Evelyn 'Champagne' King", "The Jacksons",
            "Anita Ward", "Peaches & Herb", "Van McCoy", "The Hues Corporation", "Patrick Hernandez",
            "Yvonne Elliman", "Andrea True Connection", "Heatwave", "GQ", "Instant Funk"
        ]
    },
    "hiphop": {
        "songs": [
            "The Sugarhill Gang - Rapper's Delight",
            "Grandmaster Flash & The Furious Five - The Message",
            "A Tribe Called Quest - Can I Kick It?",
            "The Notorious B.I.G. - Juicy",
            "Dr. Dre ft. Snoop Dogg - Nuthin' But A 'G' Thang",
            "Tupac - California Love",
            "Wu-Tang Clan - C.R.E.A.M.",
            "Nas - N.Y. State of Mind",
            "Public Enemy - Fight The Power",
            "Run-DMC - Walk This Way"
        ],
        "artists": [
            "The Notorious B.I.G.", "Tupac Shakur (2Pac)", "Nas", "Jay-Z", "Eminem",
            "A Tribe Called Quest", "Wu-Tang Clan", "Dr. Dre", "Snoop Dogg", "Kendrick Lamar",
            "OutKast", "Kanye West", "Lauryn Hill / The Fugees", "Public Enemy", "Run-DMC",
            "N.W.A.", "LL Cool J", "Rakim (Eric B. & Rakim)", "Grandmaster Flash", "Beastie Boys",
            "Ice Cube", "Missy Elliott", "J. Cole", "Drake", "Lil Wayne",
            "Queen Latifah", "Salt-N-Pepa", "De La Soul", "Common", "Mos Def"
        ]
    },
    "jazz": {
        "songs": [
            "Miles Davis - So What",
            "Dave Brubeck Quartet - Take Five",
            "John Coltrane - Giant Steps",
            "Louis Armstrong - What a Wonderful World",
            "Duke Ellington - Take the 'A' Train",
            "Billie Holiday - Strange Fruit",
            "Ella Fitzgerald - Summertime",
            "Thelonious Monk - 'Round Midnight",
            "Charles Mingus - Goodbye Pork Pie Hat",
            "Herbie Hancock - Cantaloupe Island"
        ],
        "artists": [
            "Miles Davis", "Louis Armstrong", "John Coltrane", "Duke Ellington", "Thelonious Monk",
            "Charlie Parker", "Dizzy Gillespie", "Billie Holiday", "Ella Fitzgerald", "Charles Mingus",
            "Herbie Hancock", "Dave Brubeck", "Count Basie", "Lester Young", "Art Tatum",
            "Nina Simone", "Sarah Vaughan", "Chet Baker", "Wes Montgomery", "Cannonball Adderley",
            "Ornette Coleman", "Sun Ra", "Art Blakey", "Max Roach", "Jaco Pastorius",
            "Stan Getz", "Benny Goodman", "Bill Evans", "Oscar Peterson", "Jelly Roll Morton"
        ]
    },
    "metal": {
        "songs": [
            "Metallica - Master of Puppets",
            "Black Sabbath - Paranoid",
            "Iron Maiden - The Trooper",
            "Slayer - Raining Blood",
            "Judas Priest - Breaking the Law",
            "Pantera - Cowboys from Hell",
            "Megadeth - Holy Wars... The Punishment Due",
            "Motörhead - Ace of Spades",
            "System Of A Down - Chop Suey!",
            "Led Zeppelin - Immigrant Song"
        ],
        "artists": [
            "Black Sabbath", "Metallica", "Iron Maiden", "Judas Priest", "Slayer",
            "Megadeth", "Pantera", "Motörhead", "Dio", "Led Zeppelin",
            "System Of A Down", "TOOL", "Death", "Ozzy Osbourne", "AC/DC",
            "Deep Purple", "Korn", "Slipknot", "Rammstein", "Guns N' Roses",
            "Venom", "Celtic Frost", "Bathory", "Mercyful Fate", "Manowar",
            "Anthrax", "Testament", "Exodus", "Opeth", "Dream Theater"
        ]
    },
    "pop": {
        "songs": [
            "Michael Jackson - Billie Jean",
            "Madonna - Like a Prayer",
            "ABBA - Dancing Queen",
            "The Beatles - I Want to Hold Your Hand",
            "Britney Spears - ...Baby One More Time",
            "Taylor Swift - Shake It Off",
            "Beyoncé - Crazy in Love",
            "Lady Gaga - Bad Romance",
            "Whitney Houston - I Wanna Dance with Somebody",
            "Prince - Kiss"
        ],
        "artists": [
            "Michael Jackson", "The Beatles", "Madonna", "Elvis Presley", "Taylor Swift",
            "Beyoncé", "Rihanna", "Lady Gaga", "Prince", "Whitney Houston",
            "ABBA", "Elton John", "Mariah Carey", "Britney Spears", "Katy Perry",
            "Adele", "Bruno Mars", "Justin Timberlake", "Janet Jackson", "Stevie Wonder",
            "Frank Sinatra", "Aretha Franklin", "Cher", "David Bowie", "George Michael",
            "Cyndi Lauper", "The Supremes", "The Beach Boys", "Spice Girls", "Frankie Valli & The Four Seasons"
        ]
    },
    "reggae": {
        "songs": [
            "Bob Marley & The Wailers - No Woman, No Cry",
            "Peter Tosh - Legalize It",
            "Jimmy Cliff - The Harder They Come",
            "Toots and the Maytals - 54-46 Was My Number",
            "Burning Spear - Marcus Garvey",
            "UB40 - Red Red Wine",
            "Steel Pulse - Ku Klux Klan",
            "Black Uhuru - Sinsemilla",
            "Desmond Dekker - Israelites",
            "Sister Nancy - Bam Bam"
        ],
        "artists": [
            "Bob Marley & The Wailers", "Peter Tosh", "Jimmy Cliff", "Toots and the Maytals", "Burning Spear",
            "Steel Pulse", "Black Uhuru", "Desmond Dekker", "Lee 'Scratch' Perry", "King Tubby",
            "UB40", "Gregory Isaacs", "Dennis Brown", "Sly & Robbie", "Horace Andy",
            "The Skatalites", "Augustus Pablo", "Max Romeo", "Culture", "Israel Vibration",
            "Yellowman", "Eek-A-Mouse", "Chronixx", "Protoje", "Damian 'Jr. Gong' Marley",
            "Stephen Marley", "Ziggy Marley", "Barrington Levy", "Ini Kamoze", "Sister Nancy"
        ]
    },
    "rock": {
        "songs": [
            "Queen - Bohemian Rhapsody",
            "Led Zeppelin - Stairway to Heaven",
            "The Rolling Stones - (I Can't Get No) Satisfaction",
            "Nirvana - Smells Like Teen Spirit",
            "AC/DC - Back in Black",
            "Pink Floyd - Another Brick in the Wall, Pt. 2",
            "The Eagles - Hotel California",
            "Guns N' Roses - Sweet Child O' Mine",
            "Chuck Berry - Johnny B. Goode",
            "The Who - My Generation"
        ],
        "artists": [
            "The Beatles", "The Rolling Stones", "Led Zeppelin", "Queen", "Pink Floyd",
            "The Who", "Jimi Hendrix", "Bob Dylan", "David Bowie", "Nirvana",
            "AC/DC", "U2", "Eagles", "Guns N' Roses", "Bruce Springsteen",
            "Tom Petty", "Creedence Clearwater Revival", "The Doors", "Fleetwood Mac", "Aerosmith",
            "Eric Clapton", "Chuck Berry", "Elvis Presley", "Buddy Holly", "The Kinks",
            "The Clash", "Ramones", "R.E.M.", "Pearl Jam", "Red Hot Chili Peppers"
        ]
    }
}