#!/usr/bin/perl 

$usage = "usage: data.txt";

$txt = shift @ARGV;
die $usage unless ($txt =~ /\.txt/);

open(FH,$txt) or die $!;

@lines = <FH>;

$xprev = $yprev = -1;

for $line (@lines) {
    chomp $line;
    next if $line =~ /^\s+\+\+\s/;
    @items = split(/,/,$line);
    # x, y, time (ms)
    ($x,$y,$time) = @items;    
    next if (!$time); #skip blank lines

    if ($x != $xprev || $y != $yprev) {
        if (@timings) {
            $av = average(\@timings);
            push @results, "$xprev, $yprev, $av\n";
        }
        @timings = ();
    }
    push @timings, $time;
    $xprev = $x;
    $yprev = $y;
}
$av = average(\@timings);
push @results, "$xprev, $yprev, $av\n";

($name = $txt) =~ s/.txt$//;
print "X, Y, $name Time(ms)\n";
print sort { $a <=> $b } @results;

sub average {    
    my @arr = @{scalar shift};
    my $sum = 0;
    for $x (@arr) {
        $sum += $x;
    }
    $len = scalar @arr;
    #print ".$sum. / .$len.\n";
    return $sum/$len if $len; #avoid divide by zero
    return 0;
}
