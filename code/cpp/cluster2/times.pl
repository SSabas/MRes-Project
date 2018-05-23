#!/usr/bin/perl
$IN = shift @ARGV;
$n = 10000;

foreach $depth (3,4) {
foreach $branch (2,4) {
foreach $method (" -SEQ ", "") {
    open IN;
    open OUT, ">tmp";
    while (<IN>) {
	if (/STAGES/) { print OUT "STAGES $depth\n"; next; }
	if (/BRANCHING/) {print OUT "BRANCHING ","$branch "x$depth,"\n";next;}
	print OUT;
    }
    close IN;
    close OUT;

    open PROC, "cluster -f tmp -n $n $method |";
    while (<PROC>) {
	if (/seconds/) {
	    print "Depth $depth, branching $branch, method ";
	    if ($method =~ /SEQ/) { print "sequential: \t"; }
	    else {                  print "parallel:   \t"; }
	    print;
	}
    }

}}}

	    
