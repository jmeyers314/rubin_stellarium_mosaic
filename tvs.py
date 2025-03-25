#! /Users/jmeyers3/src/lsstsw/miniconda/envs/lsst-scipipe-9.0.0/bin/python
# -*- coding: utf-8 -*-

import ast
from pathlib import Path
from time import sleep

import astropy.units as u
import click
import numpy as np
import requests
from astroplan import Observer
from astropy.coordinates import AltAz, Angle, EarthLocation, SkyCoord
from astropy.time import Time


DEFAULT_URL = "http://localhost:8090"
CAMERA_CHOICES = ["LSSTCam", "ComCam", "LATISS"]
DEFAULT_CONSDB_SERVER = "https://usdf-rsp.slac.stanford.edu/consdb"
DEFAULT_CONSDB_TOKEN_PATH = "~/.lsst/consdb_token"
DEFAULT_EFD_CLIENT = "usdf_efd"


def slew_to(url, camera, ra, dec, rsp):
    """Send camera repositioning command to Stellarium.

    Parameters
    ----------
    url : str
        URL of Stellarium API.
    camera : str
        Camera to slew.
    ra : float
        Right ascension in degrees.
    dec : float
        Declination in degrees.
    rsp : float
        Rotator sky position in degrees.
    """
    url = f"{url}/api/stelproperty/set"
    for data in [
        {"id": "MosaicCamera.currentCamera", "value": camera},
        {"id": "MosaicCamera.visible", "value": "True"},
        {"id": "MosaicCamera.ra", "value": ra},
        {"id": "MosaicCamera.dec", "value": dec},
        {"id": "MosaicCamera.rotation", "value": rsp},
    ]:
        requests.post(url, data=data)


def parse_angle(value, unit=u.deg):
    """Parse an angle with units.

    Parameters
    ----------
    value : str
        Angle string.
    unit : astropy.units.Unit, optional
        Unit of the angle if not inferable from value.  Default is degrees.

    Returns
    -------
    astropy.coordinates.Angle
        Angle object.
    """
    try:
        angle = Angle(value)
    except u.UnitsError:
        angle = Angle(value, unit=unit)
    return angle


def get_stellarium_attributes(url):
    """Get particular Stellarium attributes of interest.

    Parameters
    ----------
    url : str
        URL of Stellarium API.

    Returns
    -------
    dict
        Dictionary with structure:
            time: astropy.time.Time
                Current time in Stellarium.
            az: astropy.coordinates.Angle
                Current azimuth of Stellarium view.
            alt: astropy.coordinates.Angle
                Current altitude of Stellarium view.
            rsp: astropy.coordinates.Angle
                Current rotator sky position of active Stellarium camera.
    """
    time = Time(
        requests.get(f"{url}/api/main/status").json()["time"]["jday"], format="jd"
    )
    view = requests.get(f"{url}/api/main/view").json()
    azp, alt = rect_to_sph(*ast.literal_eval(view["altAz"]))
    # azp is E of S, but we want E of N.
    az = 180 * u.deg - azp
    properties = requests.get(f"{url}/api/stelproperty/list").json()
    # rsp is E of N.
    rsp = Angle(properties["MosaicCamera.rotation"]["value"], unit=u.deg)
    return dict(time=time, az=az, alt=alt, rsp=rsp)


def rect_to_sph(x, y, z):
    """Convert rectangular to spherical coordinates.

    Parameters
    ----------
    x, y, z : float
        Rectangular coordinates.

    Returns
    -------
    tuple
        Spherical coordinates (longitude, latitude).
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arcsin(z / r)
    lon = np.arctan2(y, x)
    return Angle(lon * u.rad), Angle(lat * u.rad)


@click.group()
@click.option(
    "--url",
    default=DEFAULT_URL,
    help=f"URL of Stellarium API. [default: {DEFAULT_URL}]",
)
@click.pass_context
def cli(ctx, url):
    ctx.ensure_object(dict)
    ctx.obj["URL"] = url


@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.argument("lon", type=str)
@click.argument("lat", type=str)
@click.argument("rot", required=False, type=str)
@click.option(
    "--camera",
    type=click.Choice(CAMERA_CHOICES, case_sensitive=True),
    help="Specify camera to slew. [default: LSSTCam]",
    default="LSSTCam",
)
@click.option(
    "--horizon",
    is_flag=True,
    help="Use horizon coordinates (Az/Alt/RotTelPos) instead of equitorial "
    "(RA/Dec/RotSkyPos).",
)
@click.option(
    "--time",
    type=str,
    help="Time of the slew.  Must be a string that can be parsed by astropy.time.Time. "
    "[default: current stellarium time]",
)
@click.option(
    "--timeformat",
    type=str,
    help="Format for Time.  Must be a string that can be parsed by astropy.time.Time.",
)
@click.option(
    "--no-follow",
    is_flag=True,
    help="Do not follow the slew with a Stellarium view change.",
)
@click.pass_context
def slew(ctx, lon, lat, rot, camera, horizon, time, timeformat, no_follow):
    """Slew mosaic to given position and rotation.

    LON is the longitudinal angle (RA or Az).

    LAT is the latitudinal angle (Dec or Alt).

    ROT is the rotation angle (RotSkyPos (E of N) or RotTelPos (CCW from +alt)).

    These may be specified in any format that astropy.units.Angle can parse.
    """
    api_url = ctx.obj["URL"]
    extra_args = ctx.args
    if extra_args:
        lon = extra_args[0] if len(extra_args) > 0 else lon
        lat = extra_args[1] if len(extra_args) > 1 else lat
        rot = extra_args[2] if len(extra_args) > 2 else rot

    lon = parse_angle(lon, unit=u.hour)
    lat = parse_angle(lat)

    if rot is None:
        rot = get_stellarium_attributes(api_url)["rsp"]
    else:
        rot = parse_angle(rot)

    if horizon:
        if time:
            time = Time(time, format=timeformat)
        else:
            time = get_stellarium_attributes(api_url)["time"]
        # Create SkyCoord in AltAz using lon/lat/time above
        location = EarthLocation.of_site("Cerro Pachon")
        observer = Observer(location=location, name="Rubin Observatory")
        coord = SkyCoord(alt=lat, az=lon, frame=AltAz(obstime=time, location=location))
        # Get RA/Dec from AltAz
        coord = coord.icrs
        ra = coord.ra.deg
        dec = coord.dec.deg
        # Get Parallactic Angle
        q = observer.parallactic_angle(time, coord)
        rsp = q.deg - rot.deg
    else:
        ra = lon.deg
        dec = lat.deg
        rsp = rot.deg

    slew_to(api_url, "LSSTCam" if camera == "ComCam" else camera, ra, dec, rsp)
    if not no_follow:
        requests.post(
            f"{api_url}/api/stelaction/do", data={"id": "actionSetViewToCamera"}
        )


@cli.command()
@click.argument("visit", type=int)
@click.option(
    "--camera",
    type=click.Choice(CAMERA_CHOICES, case_sensitive=True),
    help="Specify camera to slew. [default: LSSTCam]",
    default="LSSTCam",
)
@click.option(
    "--consdb-token",
    type=str,
    help=f"Path to ConsDB token file. [default: {DEFAULT_CONSDB_TOKEN_PATH}]",
    default=DEFAULT_CONSDB_TOKEN_PATH,
)
@click.option(
    "--consdb-server",
    type=str,
    help=f"ConsDB server URL. [default: {DEFAULT_CONSDB_SERVER}]",
    default=DEFAULT_CONSDB_SERVER,
)
@click.option(
    "--database",
    type=str,
    help="ConsDB database name. [default: depends on camera]",
    default=None,
)
@click.option(
    "--no-follow",
    is_flag=True,
    help="Do not follow the slew with a Stellarium view change.",
)
@click.pass_context
def visit(ctx, visit, camera, consdb_token, consdb_server, database, no_follow):
    """Slew to match given visit ID."""
    from lsst.summit.utils import ConsDbClient

    api_url = ctx.obj["URL"]

    path = Path(consdb_token).expanduser()
    with open(path, "r") as f:
        token = f.read()
    consdb_server = consdb_server.replace("https://", f"https://user:{token}@")
    consdb = ConsDbClient(consdb_server)

    if database is None:
        if camera == "ComCam":
            database = "cdb_lsstcomcam"
        elif camera == "LSSTCam":
            database = "cdb_lsstcam"
        elif camera == "LATISS":
            database = "cdb_latiss"

    data = consdb.query(f"select * from {database}.visit1 where visit_id = {visit}")[0]

    slew_to(
        api_url,
        "LSSTCam" if camera == "ComCam" else camera,
        data["s_ra"],
        data["s_dec"],
        data["sky_rotation"],
    )
    # Set visit time and pause with timerate=0
    requests.post(
        f"{api_url}/api/main/time",
        data={"time": data["exp_midpt_mjd"] + 2400000.5, "timerate": 0},
    )
    if not no_follow:
        requests.post(
            f"{api_url}/api/stelaction/do", data={"id": "actionSetViewToCamera"}
        )


@cli.command()
@click.argument("day_obs0", type=int)
@click.argument("day_obs1", type=int, required=False)
@click.option(
    "--camera",
    type=click.Choice(CAMERA_CHOICES, case_sensitive=True),
    help="Specify camera to replay. [default: LSSTCam]",
    default="LSSTCam",
)
@click.option(
    "--consdb-token",
    type=str,
    help=f"Path to ConsDB token file. [default: {DEFAULT_CONSDB_TOKEN_PATH}]",
    default=DEFAULT_CONSDB_TOKEN_PATH,
)
@click.option(
    "--consdb-server",
    type=str,
    help=f"ConsDB server URL. [default: {DEFAULT_CONSDB_SERVER}]",
    default=DEFAULT_CONSDB_SERVER,
)
@click.option(
    "--database",
    type=str,
    help="ConsDB database name. [default: Depends on camera]",
    default=None,
)
@click.option(
    "--no-follow",
    is_flag=True,
    help="Do not follow the replay with a Stellarium view change.",
)
@click.pass_context
def replay(
    ctx, day_obs0, day_obs1, camera, consdb_token, consdb_server, database, no_follow
):
    """Replay a range of days.

    DAY_OBS0 and DAY_OBS1 form a range of days (inclusive) to replay.  If DAY_OBS1 is
    omitted, it is set equal to DAY_OBS0.

    """
    from lsst.summit.utils import ConsDbClient

    api_url = ctx.obj["URL"]

    path = Path(consdb_token).expanduser()
    with open(path, "r") as f:
        token = f.read()
    consdb_server = consdb_server.replace("https://", f"https://user:{token}@")
    consdb = ConsDbClient(consdb_server)

    if database is None:
        if camera == "ComCam":
            database = "cdb_lsstcomcam"
        elif camera == "LSSTCam":
            database = "cdb_lsstcam"
        elif camera == "LATISS":
            database = "cdb_latiss"

    if day_obs1 is None:
        day_obs1 = day_obs0

    visits = consdb.query(
        f"select * from {database}.visit1 where day_obs >= {day_obs0} and day_obs <= {day_obs1}"
    )

    # Set stellarium time to first visit with img_type either OBJECT or ACQ
    visit0 = visits[
        np.logical_or(visits["img_type"] == "OBJECT", visits["img_type"] == "ACQ")
    ][0]
    requests.post(
        f"{api_url}/api/main/time",
        data={"time": visit0["exp_midpt_mjd"] + 2400000.5, "timerate": 100 / 86400},
    )

    currentindex = None
    while True:
        time = get_stellarium_attributes(api_url)["time"]
        index = np.searchsorted(visits["exp_midpt_mjd"], time.mjd)
        if index == currentindex:
            continue
        currentindex = index
        try:
            visit = visits[index]
        except IndexError:
            continue
        print("\n\n\n")
        print(
            visit[
                [
                    "visit_id",
                    "day_obs",
                    "seq_num",
                    "band",
                    "exp_time",
                    "img_type",
                    "science_program",
                    "observation_reason",
                    "target_name",
                ]
            ]
        )
        slew_to(
            api_url,
            "LSSTCam" if camera == "ComCam" else camera,
            visit["s_ra"],
            visit["s_dec"],
            visit["sky_rotation"],
        )
        if not no_follow:
            requests.post(
                f"{api_url}/api/stelaction/do", data={"id": "actionSetViewToCamera"}
            )


def query_mount_status(efd_client, topic, time):
    """Query the mount status from the EFD.

    Parameters
    ----------
    efd_client : lsst_efd_client.EfdClient
        EFD client.
    topic : str
        Topic to query.
    time : astropy.time.Time
        Time to query.

    Returns
    -------
    dict
        Mount status.
    """

    from lsst.summit.utils.efdUtils import getMostRecentRowWithDataBefore

    while True:
        try:
            mount = getMostRecentRowWithDataBefore(
                efd_client, topic, timeToLookBefore=time, maxSearchNMinutes=5
            )
        except (TypeError, ValueError):
            sleep(1.0)
            print("Waiting for mount status...")
            continue
        else:
            break
    return mount


@cli.command()
@click.argument(
    "camera",
    type=click.Choice(CAMERA_CHOICES, case_sensitive=True),
    default="LSSTCam",
    metavar="CAMERA",
)
@click.option(
    "--efd-client",
    type=str,
    help=f"EFD client name. [default: {DEFAULT_EFD_CLIENT}]",
    default=DEFAULT_EFD_CLIENT,
)
@click.option(
    "--delay",
    type=float,
    help="Delay between updates in seconds. [default: 1]",
    default=1.0,
)
@click.option(
    "--no-follow",
    is_flag=True,
    help="Do not follow the replay with a Stellarium view change.",
)
@click.pass_context
def follow(ctx, camera, efd_client, delay, no_follow):
    """Follow camera pointing in real time with Stellarium.

    CAMERA is the camera to follow (LSSTCam, ComCam, or LATISS).
    """
    from lsst_efd_client import EfdClient

    api_url = ctx.obj["URL"]

    efd_client = EfdClient(efd_client)

    if camera in ["ComCam", "LSSTCam"]:
        topic = "lsst.sal.MTPtg.mountStatus"
    elif camera == "LATISS":
        topic = "lsst.sal.ATPtg.mountStatus"
    else:
        raise ValueError(f"Unknown camera: {camera}")

    pachon = Observer.at_site("Cerro Pachon")

    while True:
        tnow = Time.now()
        mount_status = query_mount_status(efd_client, topic, tnow)
        ra = mount_status["mountRA"] * 15
        dec = mount_status["mountDec"]
        rtp = mount_status["mountRot"]
        tefd = Time(mount_status["timestamp"], format="unix")
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        parallactic_angle = pachon.parallactic_angle(tefd, coord)
        rsp = parallactic_angle.deg - rtp
        if rsp < 0:
            rsp += 360
        if rsp > 360:
            rsp -= 360

        print(f"RA: {ra}, Dec: {dec}, Rot: {rtp}, RSP: {rsp}")

        slew_to(api_url, "LSSTCam" if camera == "ComCam" else camera, ra, dec, rsp)
        if not no_follow:
            requests.post(
                f"{api_url}/api/stelaction/do", data={"id": "actionSetViewToCamera"}
            )
        sleep(delay)


if __name__ == "__main__":
    cli()
