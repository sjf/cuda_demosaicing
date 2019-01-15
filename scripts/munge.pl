#!/usr/bin/perl -w
sub av {
    $r = shift;
    $n = $sum = 0;
    for $i (@{$r}) {
        $sum += $i;
        $n++;
    }
    $av = $sum / $n;
    $dev = 0;
    for $i (@{$r}) {
        $dev += ($i - $av)**2;
    }
    $dev = sqrt($dev);
    return ($sum/$n,$dev);
}

for $f (@ARGV) {
    open( F, $f) or die "Cannot open $f $!";
    print $f;
    <F>; # skip header
    while (<F>) {
        ($func,$file,$x,$y,$pixels,$rt,$ut,$st,$mask,$total) = split /,\s*/;
        $total = $ut + $st;
        $r = $vals{$pixels};
        %res = ($r) ? %{$r} : (); 
        push @{$res{$f}}, $total;
        $vals{$pixels} = \%res;
    }
}

print "# No. pixels, Average Time, Standard Deviationn\n";
foreach $p (keys %vals) {
    %res = %{$vals{$p}};
    foreach $f (keys %res) {
        #print "$p -> $f\n";
        @times = @{$res{$f}};
        
        ($m,$dev) = av(\@times);
        print "$p -> $f -> |@times|\n";
    }


    #print "$p -> @{$vals{$p}} -> $m, $dev\n";
    #print "$p, $m, $dev\n";
    #print "$p $m $dev\n";
    #}
}

