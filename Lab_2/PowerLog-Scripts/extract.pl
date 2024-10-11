#!perl -w
# this is a perl script

# a PERL script to extract power and energy figues from PowerLog3.0
#
#

#------------- MAIN -------------
#use strict; 
#use warnings;

if(@ARGV == 0) {
 #   die "usage: extract [<file>]\n";
}

$input_file = $ARGV[0];

$output_file1 = "report.csv";
open(STUFF2, "> $output_file1") || die "Cannot open $output_file1 for writing.";
print STUFF2 "Opt.,Exec. Time(s),Average Freq. (GHz),Average Power Dissipation(W),Energy Consumption (J) ,Temperature (C)\n";

my @inputs = ("2mm_o0", "2mm_o1", "2mm_o2", "2mm_o3", "2mm_ofast","2mm_os", "2mm_o0_omp", "2mm_o1_omp", "2mm_o2_omp", "2mm_o3_omp", "2mm_ofast_omp","2mm_os_omp");


foreach $name ( @inputs ) {

$input_file = $name.".csv";


open(STUFF1, $input_file) || die "Cannot open $input_file for read.";

print "Extracting from file...$input_file\n";

$start_vals = 0;
$max_freq = 0;
$min_freq = 5000;
$sum_freq = 0;
$average_freq=0;

$max_temp = 0;
$min_temp = 5000;
$sum_temp = 0;
$average_temp=0;

$num_measurements=0;

while($line=<STUFF1>) {
	
	if($start_vals == 1) {
		if($line =~ m/^([^,]+,){4}([^,]+),([^,]+,){6}([^,]+).*/) {
			$Freq=$2;
			#print "---- $Freq (GHz)\n";
			#print "---- $line\n";
			if($Freq > $max_freq) {$max_freq = $Freq;}
			if($Freq < $min_freq) {$min_freq = $Freq;}
			$num_measurements++;
			$sum_freq = $sum_freq + $Freq;
			
			$Temp=$4;
			if($Temp > $max_temp) {$max_temp = $Temp;}
			if($Temp < $min_temp) {$min_temp = $Temp;}

			$sum_temp = $sum_temp + $Temp;
		} else {
			$start_vals = 0;
			print "END row of values\n";
		}
	}
	if($line =~ m/^System Time/) {
		$start_vals = 1;
		print "BEGIN row of values\n";
	}
 
	#if($line =~ m/^Measured\sRDTSC\sFrequency\s\(GHz\)\s=\s(\d+\.\d+)/) {
	#  $Freq=$1;
	#  print "---- $Freq (GHz)\n";
	#} 
  if($line =~ m/^Average\sProcessor\sPower_0\s\(Watt\)\s=\s(\d+\.\d+)/) {
    $Power=$1;
    print "---- $Power (W)\n";
  } 
  elsif($line =~ m/^Cumulative\sProcessor\sEnergy_0\s\(Joules\)\s=\s(\d+\.\d+)/) {
    $Energy=$1;
    print "---- $Energy (J)\n";
  } 
  elsif($line =~ m/^Total\sElapsed\sTime\s\(sec\)\s\=\s(\d+\.\d+)/) {
    $TotalExec=$1;
    print "---- $TotalExec (sec)\n";
  } 
	#Measured RDTSC Frequency (GHz) = 1.992
	#Cumulative Processor Energy_0 (Joules) = 146.988464
	#Cumulative Processor Energy_0 (mWh) = 40.830129
	#Average Processor Power_0 (Watt) = 4.656026  
}

print "---- Number of measurements: $num_measurements\n";

$average_freq = $sum_freq/$num_measurements;
$max_freq=$max_freq/1000;
$min_freq=$min_freq/1000;
$average_freq=$average_freq/1000;

print "---- $max_freq (GHz)\n";
print "---- $min_freq (GHz)\n";
print "---- $average_freq (GHz)\n";

$average_temp = $sum_temp/$num_measurements;

print "---- $max_temp (C)\n";
print "---- $min_temp (C)\n";
print "---- $average_temp (C)\n";

close(STUFF1);
print STUFF2 "$name,$TotalExec,$average_freq,$Power,$Energy,$average_temp\n";

}

close(STUFF2);

exit(1);
#------------- END MAIN -------------