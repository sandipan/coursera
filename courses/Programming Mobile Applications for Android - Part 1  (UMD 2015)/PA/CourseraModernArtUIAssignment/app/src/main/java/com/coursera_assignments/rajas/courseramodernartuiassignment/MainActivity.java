package com.coursera_assignments.rajas.courseramodernartuiassignment;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.RelativeLayout;
import android.widget.SeekBar;

import com.coursera_assignments.rajas.courseramodernartuiassignment.R;
import com.coursera_assignments.rajas.courseramodernartuiassignment.MoreInfo;
import com.coursera_assignments.rajas.courseramodernartuiassignment.FragmentDialog;

public class MainActivity extends Activity implements FragmentDialog {

    private static final String TAG = MainActivity.class.getName();

    private RelativeLayout palette;

    @Override
    protected void onCreate ( Bundle savedInstanceState ) {

        super.onCreate( savedInstanceState );
        setContentView( R.layout.activity_main );

        palette = ( RelativeLayout ) findViewById( R.id.palette );
        SeekBar seek = ( SeekBar ) findViewById( R.id.seekBar );

        seek.setOnSeekBarChangeListener( new SeekBar.OnSeekBarChangeListener() {

            @Override
            public void onProgressChanged ( SeekBar seekBar, int progress, boolean fromUser ) {

                for ( int i = 0; i < palette.getChildCount(); i++ ) {
                    View child = palette.getChildAt( i );

                    int originalColor = Color.parseColor( ( String ) child.getTag() );
                    int invertedColor = ( 0x00FFFFFF - ( originalColor | 0xFF000000 ) ) |
                            ( originalColor & 0xFF000000 );

                    if ( getResources().getColor( R.color.white ) != originalColor &&
                            getResources().getColor( R.color.light_gray ) != originalColor ) {

                        int origR = ( originalColor >> 16 ) & 0x000000FF;
                        int origG = ( originalColor >> 8 ) & 0x000000FF;
                        int origB = originalColor & 0x000000FF;

                        int invR = ( invertedColor >> 16 ) & 0x000000FF;
                        int invG = ( invertedColor >> 8 ) & 0x000000FF;
                        int invB = invertedColor & 0x000000FF;

                        child.setBackgroundColor( Color.rgb(
                                ( int ) ( origR + ( invR - origR ) * ( progress / 100f ) ),
                                ( int ) ( origG + ( invG - origG ) * ( progress / 100f ) ),
                                ( int ) ( origB + ( invB - origB ) * ( progress / 100f ) ) ) );
                        child.invalidate();
                    }
                }
            }

            @Override
            public void onStartTrackingTouch ( SeekBar seekBar ) {

            }

            @Override
            public void onStopTrackingTouch ( SeekBar seekBar ) {

            }
        } );
    }

    @Override
    public boolean onCreateOptionsMenu ( Menu menu ) {

        getMenuInflater().inflate( R.menu.menu_main, menu );
        return true;
    }

    public void showDialog ( MenuItem item ) {

        new MoreInfo().show( getFragmentManager(), TAG );
    }

    public void doPositiveClick() {

        Intent visit = new Intent( Intent.ACTION_VIEW, Uri.parse("https://www.MoMA.org") );
        Intent chooser = Intent.createChooser( visit, getResources().getString( R.string.open_with ) );
        startActivity(chooser);
    }

    public void doNegativeClick()
    {

    }
}
