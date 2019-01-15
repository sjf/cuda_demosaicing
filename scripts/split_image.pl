#!/usr/bin/perl

$usage = "usage: data.txt image.ppm";

$txt = shift @ARGV;
die $usage unless ($txt =~ /\.txt/);

$img = shift @ARGV;
die $usage unless ($img =~ /\.ppm/);

(($img_base = $img) =~ s/\d*\.ppm//);

open(FH,$txt) or die $!;

@lines = <FH>;
shift @lines;

for $line (@lines) {
    chomp $line;
    @items = split(/\t/,$line);
    #Image No. X Y no. pixels (desired) ratio actual no. pixels no. pixels error Factor Xoffset Yoffset X rounded
    ($n,$x,$y) = @items;
    $xoff = $items[8];
    $yoff = $items[9];
    $items[10] =~ /(\d+)/;
    $xround = $1;
    $n = "0$n" if ($n < 10);
    $cmd = "convert -crop ${xround}x${y}+${xoff}+${yoff} -depth 8 _$img $img_base${n}.ppm";
    print "$cmd\n";
    print `$cmd`;
}
