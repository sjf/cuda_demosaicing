#!/usr/bin/perl
use POSIX qw(ceil floor);
$FACTOR=500000;
$WB=4320;
%WH=2880;

$usage = "usage: image.ppm";

$img = shift @ARGV;
die $usage unless ($img =~ /\.ppm/);

#$img_base = "${img}_";
(($img_base = $img) =~ s/\d*\.ppm//);

$ident = `identify $img`;
if ($ident =~ /(\d+)x(\d+)/) {
    $x = $x0 = $1;
    $y = $y0 = $2;
} else {
    die "Could not get image size\n";
}

#$x = $x0 = 4290;#shift;
#$y = $y0 = 2856;#shift;
#if (abs($x/$y - 1.5) > 0.01) {
#    $ratio = $x/$y;
#    print "Warning ratio is $ratio\n";
#} else {
    $ratio = 1.5;
#}

$i = 1;
$xoffset = 0;
$yoffset = 0;
$xround = round($x,32);
$num_pixels = $x * $y;

$n = ($i < 10) ? "0$i" : "$i";

#print "$n, $x, $xround, $y, $num_pixels, $xoffset, $yoffset, $ratio\n";
$name = "_${img_base}_${n}.ppm";
if (! -f $name){
 $cmd = "convert -crop ${xround}x${y}+${xoffset}+${yoffset} -depth 8 $img $name";
 print "$cmd\n";
 print `$cmd`;
}
while ($i++ < 25) {
    $num_pix_desired = $num_pixels - $FACTOR;
    last if ($num_pix_desired <= 0);
    #$ratio = $x/$y;
    $y = floor(sqrt($num_pix_desired/$ratio));
    $x = floor($y * $ratio);
    $num_pixels = $x * $y;
    $xround = round($x,32);
    $xoffset = round(($x0 - $xround)/2,4);
    $yoffset = round(($y0 - $y)/2,4);

    if ($xround + $xoffset > $x0) {
        print "too much x\n";
    } elsif ($y + $yoffset > $y0) {
        print "too much y\n";
    }

    $n = ($i < 10) ? "0$i" : "$i";
    $name = "_${img_base}_${n}.ppm";   
    if (! -f $name) {
      #print "$n, $x, $xround, $y, $num_pixels, $xoffset, $yoffset, $ratio\n";
      $cmd = "convert -crop ${xround}x${y}+${xoffset}+${yoffset} -depth 8 $img $name";
      print "$cmd\n";
      print `$cmd`;
   }

}

sub round() {
    my $n = shift;
    my $base = shift;
    return floor($n / $base) * $base;
}
